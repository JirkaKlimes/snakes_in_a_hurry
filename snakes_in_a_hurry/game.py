import uuid
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import numba
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32


@dataclass
class GameState:
    head_positions: np.ndarray | DeviceNDArray
    food_position: np.ndarray | DeviceNDArray
    body_directions: np.ndarray | DeviceNDArray
    body_direction_start_offset: np.ndarray | DeviceNDArray
    body_lengths: np.ndarray | DeviceNDArray
    bitboards: np.ndarray | DeviceNDArray
    is_dead: np.ndarray | DeviceNDArray
    tail_positions: np.ndarray | DeviceNDArray


@dataclass
class ColorConfig:
    head: np.ndarray = field(
        default_factory=lambda: np.uint8([[0, 87, 224], [224, 87, 0]])
    )
    body: np.ndarray = field(
        default_factory=lambda: np.uint8([[96, 155, 247], [247, 155, 96]])
    )
    food: np.ndarray = field(default_factory=lambda: np.uint8([32, 224, 0]))
    empty: np.ndarray = field(default_factory=lambda: np.uint8([30, 30, 30]))
    border: np.ndarray = field(default_factory=lambda: np.uint8([60, 60, 60]))


class CUDASnakeGame:
    def __init__(self, size: int, num: int, color_config: Optional[ColorConfig] = None):
        assert (
            not size & 3 and 256 >= size > 4
        ), "Game size must be multiple of 4 and in range (8, 255)"

        self.color_config = color_config if color_config else ColorConfig()

        self.size = size
        self.num = num

        self.DIR_OFFSETS = np.int8([[0, -1], [0, 1], [-1, 0], [1, 0]])
        self.INVERSE_MOVES = np.uint8([1, 0, 3, 2])

        self.game_state_host = GameState(
            head_positions=np.empty((self.num, 2, 2), dtype=np.uint8),
            food_position=np.empty((self.num, 2), dtype=np.uint8),
            body_directions=np.empty(
                (self.num, 2, (self.size * self.size) // 8), dtype=np.uint16
            ),
            body_direction_start_offset=np.empty((self.num, 2), dtype=np.uint16),
            body_lengths=np.empty((self.num, 2), dtype=np.uint16),
            bitboards=np.empty(
                (self.num, self.size // 4, self.size // 4, 2), dtype=np.uint16
            ),
            is_dead=np.empty((self.num, 2), dtype=np.uint8),
            tail_positions=np.empty((self.num, 2, 2), dtype=np.uint8),
        )
        self.image_device: DeviceNDArray = None
        self.num_image_device = 0

    def reset_games(self):
        spawn_offset = self.size // 4

        initial_head_positions = np.uint8(
            [
                [self.size - spawn_offset - 1, spawn_offset],
                [spawn_offset, self.size - spawn_offset - 1],
            ]
        )
        self.game_state_host.head_positions[:] = initial_head_positions
        self.game_state_host.tail_positions[:] = np.uint8(
            [
                [self.size - spawn_offset, spawn_offset - 1],
                [spawn_offset - 1, self.size - spawn_offset],
            ]
        )

        body_directions = np.zeros((2, self.size * self.size), dtype=np.uint16)
        body_directions[:, :2] = np.array([[0, 3], [1, 2]])
        shifts = np.tile(
            np.arange(0, 16, 2, dtype=np.uint16), (self.size * self.size) // 8
        )
        body_directions <<= shifts
        body_directions = np.reshape(
            body_directions, (2, (self.size * self.size) // 8, 8)
        ).sum(axis=-1)
        self.game_state_host.body_directions[:] = body_directions
        self.game_state_host.body_lengths[...] = 2
        self.game_state_host.body_direction_start_offset[...] = 0
        self.game_state_host.food_position[:] = np.random.randint(
            self.size // 2 - 1,
            self.size // 2 + 1,
            size=self.game_state_host.food_position.shape,
        )
        self.game_state_host.is_dead[...] = 0

        bitboards = np.zeros((2, self.size, self.size), dtype=np.uint16)
        bitboards[0, initial_head_positions[0, 1] - 1, initial_head_positions[0, 0]] = 1
        bitboards[
            0, initial_head_positions[0, 1] - 1, initial_head_positions[0, 0] + 1
        ] = 1
        bitboards[1, initial_head_positions[1, 1] + 1, initial_head_positions[1, 0]] = 1
        bitboards[
            1, initial_head_positions[1, 1] + 1, initial_head_positions[1, 0] - 1
        ] = 1
        shifts = np.tile(
            np.arange(16, dtype=np.uint16).reshape((4, 4)),
            (self.size // 4, self.size // 4),
        )
        bitboards = bitboards << shifts
        bitboards = np.reshape(
            bitboards,
            (2, self.size // 4, 4, self.size // 4, 4),
        ).sum(axis=(2, 4))
        bitboards = np.transpose(bitboards, (1, 2, 0))
        self.game_state_host.bitboards[:] = bitboards

    def sync_to_device(self):
        ndarrays = list(self.game_state_host.__dict__.values())
        device_ndarrays = map(cuda.to_device, ndarrays)
        self.game_state_device = GameState(*device_ndarrays)

    def sync_to_host(self):
        for device_arr, host_arr in zip(
            self.game_state_device.__dict__.values(),
            self.game_state_host.__dict__.values(),
        ):
            host_arr[:] = device_arr.copy_to_host()

    def prepare_cuda(self, rng_seed: int = None):
        ndarrays = list(self.game_state_host.__dict__.values())
        device_ndarrays = map(cuda.to_device, ndarrays)
        self.game_state_device = GameState(*device_ndarrays)

        rng_seed = rng_seed if rng_seed else uuid.uuid4().int & (1 << 64) - 1
        self.rng_states_device = create_xoroshiro128p_states(1, rng_seed)

        device = cuda.get_current_device()
        self.warp_size = device.WARP_SIZE
        device_name = device.name.decode()
        print(f"CUDA snakes using `{device_name}`")

        INVERSE_MOVES = self.INVERSE_MOVES
        DIR_OFFSETS = self.DIR_OFFSETS
        HEAD_COLOR = self.color_config.head
        BODY_COLOR = self.color_config.body
        FOOD_COLOR = self.color_config.food
        BORDER_COLOR = self.color_config.border
        EMPTY_COLOR = self.color_config.empty

        @cuda.jit
        def kernel_increment_games(
            many_head_pos,
            many_tail_pos,
            many_body_len,
            many_body_dir_start,
            many_body_dir,
            many_food_pos,
            many_bitboards,
            many_dead,
            many_moves,
            num_games,
            size,
            rng_states,
        ):
            inverse_moves = cuda.const.array_like(INVERSE_MOVES)
            dir_offsets = cuda.const.array_like(DIR_OFFSETS)

            game_idx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            player = cuda.threadIdx.y

            if game_idx >= num_games:
                cuda.syncthreads()
                cuda.syncthreads()
                return

            head_positions = many_head_pos[game_idx]
            tail_positions = many_tail_pos[game_idx]
            body_lengths = many_body_len[game_idx]
            body_direction_start_offset = many_body_dir_start[game_idx]
            body_direction = many_body_dir[game_idx]
            food_position = many_food_pos[game_idx]
            bitboards = many_bitboards[game_idx]
            is_dead = many_dead[game_idx]
            moves = many_moves[game_idx]

            s = body_direction_start_offset[player]
            last_directions = body_direction[player, s // 8] >> (2 * (s % 8)) & 0b11
            invalid_move = last_directions == moves[player]
            __shared_new_head_positions = cuda.shared.array(
                (32, 2, 2), dtype=numba.uint8
            )
            new_head_positions = __shared_new_head_positions[cuda.threadIdx.x]

            if invalid_move:
                moves[player] = inverse_moves[last_directions]
            new_head_positions[player, 0] = (
                head_positions[player, 0] + dir_offsets[moves[player], 0]
            )
            new_head_positions[player, 1] = (
                head_positions[player, 1] + dir_offsets[moves[player], 1]
            )

            new_head = new_head_positions[player]
            out_of_bounds = not (0 <= new_head[0] < size and 0 <= new_head[1] < size)
            if out_of_bounds:
                is_dead[player] = True

            cuda.syncthreads()

            both_alive = not (is_dead[0] or is_dead[1])
            if both_alive:
                heads_collided = (
                    new_head_positions[0, 0] == new_head_positions[1, 0]
                    and new_head_positions[0, 1] == new_head_positions[1, 1]
                )
                if heads_collided:
                    is_dead[player] = True

            food_eaten = cuda.local.array((2,), numba.boolean)
            food_eaten[0] = (
                not is_dead[player]
                and new_head_positions[player, 0] == food_position[0]
                and new_head_positions[player, 1] == food_position[1]
            )
            food_eaten[1] = (
                not is_dead[1 - player]
                and new_head_positions[1 - player, 0] == food_position[0]
                and new_head_positions[1 - player, 1] == food_position[1]
            )
            new_food_required = food_eaten[0] or food_eaten[1]

            if food_eaten[0]:
                body_lengths[player] += 1
                new_food_required = True
            elif not is_dead[player]:
                x, y = tail_positions[player]
                bitboards[y // 4, x // 4, player] &= ~(1 << (x % 4 + y % 4 * 4))
                s = (
                    body_direction_start_offset[player] + body_lengths[player] - 1
                ) % body_direction[player].shape[0]
                tail_dir = body_direction[player, s // 8] >> (2 * (s % 8)) & 0b11
                tail_dir = INVERSE_MOVES[tail_dir]
                tail_positions[player, 0] = (
                    tail_positions[player, 0] + DIR_OFFSETS[tail_dir, 0]
                )
                tail_positions[player, 1] = (
                    tail_positions[player, 1] + DIR_OFFSETS[tail_dir, 1]
                )

            if not is_dead[player]:
                body_direction_start_offset[player] = (
                    body_direction_start_offset[player] - 1
                ) % body_direction[player].shape[0]
                body_direction[player][body_direction_start_offset[player] // 8] &= ~(
                    0b11 << (2 * (body_direction_start_offset[player] % 8))
                )
                body_direction[player][body_direction_start_offset[player] // 8] |= (
                    INVERSE_MOVES[moves[player]]
                    << (2 * (body_direction_start_offset[player] % 8))
                )
                x, y = head_positions[player]
                bitboards[y // 4, x // 4, player] |= 1 << (x % 4 + y % 4 * 4)

                head_positions[player, 0] = new_head[0]
                head_positions[player, 1] = new_head[1]

            cuda.syncthreads()

            x, y = head_positions[player]
            occupied_by_0 = (
                bitboards[y // 4, x // 4, 0] >> (x % 4 + y % 4 * 4) & 0b1
                and not is_dead[0]
            )
            occupied_by_1 = (
                bitboards[y // 4, x // 4, 1] >> (x % 4 + y % 4 * 4) & 0b1
                and not is_dead[1]
            )
            if occupied_by_0 or occupied_by_1:
                is_dead[player] = True

            if new_food_required and player:
                empty_cells = int(size) ** 2
                if not is_dead[0]:
                    empty_cells -= 1 + body_lengths[0]
                if not is_dead[1]:
                    empty_cells -= 1 + body_lengths[1]
                if not empty_cells:
                    is_dead[:] = True
                    return

                food_empty_idx = int(
                    xoroshiro128p_uniform_float32(rng_states, 0) * (empty_cells - 1)
                )
                empty_seen = 0
                food_idx = 0
                while True:
                    x = food_idx % size
                    y = food_idx // size
                    occupied_by_0 = (
                        bitboards[y // 4, x // 4, 0] >> (x % 4 + y % 4 * 4) & 0b1
                        and not is_dead[0]
                    )
                    occupied_by_1 = (
                        bitboards[y // 4, x // 4, 1] >> (x % 4 + y % 4 * 4) & 0b1
                        and not is_dead[1]
                    )
                    if (not occupied_by_0) and (not occupied_by_1):
                        empty_seen += 1
                    if empty_seen == (food_empty_idx + 1):
                        break
                    food_idx += 1
                food_position[0] = food_idx % size
                food_position[1] = food_idx // size

        @cuda.jit
        def kernel_render_games(
            image: np.ndarray,
            many_head_pos: np.ndarray,
            many_body_len: np.ndarray,
            many_body_dir_start: np.ndarray,
            many_body_dir: np.ndarray,
            many_food_pos: np.ndarray,
            many_dead: np.ndarray,
            num_games: int,
            size: int,
        ):
            dir_offsets = cuda.const.array_like(DIR_OFFSETS)
            food_color = cuda.const.array_like(FOOD_COLOR)
            padding_color = cuda.const.array_like(BORDER_COLOR)
            head_color = cuda.const.array_like(HEAD_COLOR)
            body_color = cuda.const.array_like(BODY_COLOR)
            empty_color = cuda.const.array_like(EMPTY_COLOR)

            game_idx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
            player = cuda.threadIdx.y

            head_pos = many_head_pos[game_idx]
            body_lengths = many_body_len[game_idx]
            body_direction_start_offsets = many_body_dir_start[game_idx]
            body_direction = many_body_dir[game_idx]
            food_posistion = many_food_pos[game_idx]
            is_dead = many_dead[game_idx]

            grid_size = int(num_games**0.5)

            grid_y = game_idx // grid_size
            grid_x = game_idx % grid_size

            from_x = 0 if player else size // 2
            to_x = size // 2 if player else size
            for x in range(from_x, to_x):
                for y in range(size):
                    for c in range(3):
                        image[grid_y * (size + 1) + y, grid_x * (size + 1) + x, c] = (
                            empty_color[c]
                        )
            cuda.syncthreads()

            if player:
                for i in range(size):
                    for c in range(3):
                        image[
                            grid_y * (size + 1) + size, grid_x * (size + 1) + i, c
                        ] = padding_color[c]
            else:
                for i in range(size):
                    for c in range(3):
                        image[
                            grid_y * (size + 1) + i, grid_x * (size + 1) + size, c
                        ] = padding_color[c]

            if player:
                for c in range(3):
                    image[
                        grid_y * (size + 1) + food_posistion[1],
                        grid_x * (size + 1) + food_posistion[0],
                        c,
                    ] = food_color[c]

            if is_dead[player]:
                return

            for c in range(3):
                image[
                    grid_y * (size + 1) + head_pos[player, 1],
                    grid_x * (size + 1) + head_pos[player, 0],
                    c,
                ] = head_color[player, c]

            hx, hy = head_pos[player]
            for i in range(
                body_direction_start_offsets[player],
                body_direction_start_offsets[player] + body_lengths[player],
            ):
                direction = (
                    body_direction[player, (i % body_direction.shape[1]) // 8]
                    >> (2 * (i % body_direction.shape[1] % 8))
                    & 0b11
                )
                offset = dir_offsets[direction]
                hx += offset[0]
                hy += offset[1]

                for c in range(3):
                    image[grid_y * (size + 1) + hy, grid_x * (size + 1) + hx, c] = (
                        body_color[player, c]
                    )

        self.kernel_increment_games = kernel_increment_games
        self.kernel_render_games = kernel_render_games

    def increment_games_device(self, moves: DeviceNDArray):
        grid_dim = (int(np.ceil(self.num / self.warp_size)), 1)
        block_dim = (self.warp_size, 2)
        self.kernel_increment_games[grid_dim, block_dim](
            self.game_state_device.head_positions,
            self.game_state_device.tail_positions,
            self.game_state_device.body_lengths,
            self.game_state_device.body_direction_start_offset,
            self.game_state_device.body_directions,
            self.game_state_device.food_position,
            self.game_state_device.bitboards,
            self.game_state_device.is_dead,
            moves,
            np.uint32(self.num),
            np.uint32(self.size),
            self.rng_states_device,
        )

    def get_image_from_device(self, num: int, copy_to_host: bool = True):
        if num != self.num_image_device:
            self.num_image_device = num
            grid_size = int(num**0.5) * (self.size + 1)
            image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
            device_image = cuda.to_device(image)
            self.image_device = device_image

        grid_dim = (int(np.ceil(num / self.warp_size)), 1)
        block_dim = (self.warp_size, 2)
        self.kernel_render_games[grid_dim, block_dim](
            self.image_device,
            self.game_state_device.head_positions,
            self.game_state_device.body_lengths,
            self.game_state_device.body_direction_start_offset,
            self.game_state_device.body_directions,
            self.game_state_device.food_position,
            self.game_state_device.is_dead,
            np.uint32(num),
            np.uint32(self.size),
        )
        if copy_to_host:
            image = self.image_device.copy_to_host()
            return image
        return self.image_device


if __name__ == "__main__":
    import cv2 as cv
    import time

    GAME_SIZE = 8
    RENDER_GAMES = int(64**2)
    NUM_GAMES = RENDER_GAMES
    # NUM_GAMES = 8388608
    # NUM_GAMES = 4194304
    # NUM_GAMES = 2097152
    # NUM_GAMES = 1048576
    # NUM_GAMES = 65536
    # NUM_GAMES = 25600
    # NUM_GAMES = 4096
    # NUM_GAMES = 1024
    # NUM_GAMES = 100
    # NUM_GAMES = 16
    # NUM_GAMES = 4

    game = CUDASnakeGame(GAME_SIZE, NUM_GAMES)
    game.prepare_cuda()

    while True:
        print("Starting game!")
        game.reset_games()
        game.sync_to_device()

        while games_running := NUM_GAMES - np.sum(
            np.all(game.game_state_host.is_dead, axis=-1)
        ):
            st = time.monotonic()
            moves = np.random.randint(0, 4, size=(game.num, 2)).astype(np.uint8)
            device_moves = cuda.to_device(moves)
            game.increment_games_device(device_moves)

            if RENDER_GAMES:
                image = game.get_image_from_device(RENDER_GAMES)
                cv.imshow("image", image)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    quit()

            game.game_state_host.is_dead = game.game_state_device.is_dead.copy_to_host()
            print(
                f"Games still running: {games_running}, Frame time: {(time.monotonic() - st)*1e3:.5f}ms"
            )
