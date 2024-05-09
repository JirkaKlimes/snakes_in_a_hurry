from tqdm import tqdm
from snakes_in_a_hurry.game import CUDASnakeGame
import cupy as cp
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import pickle


class GameRecorder:
    def __init__(self, game: CUDASnakeGame):
        self.game = game

    def initialize_game_recording(self):
        self._bitboards = []
        self._food_positions = []
        self._head_positions = []
        self._moves = []
        self._move_counts = np.zeros((self.game.num, 2), dtype=np.uint32)

    def record_game_step(self, moves):
        self._bitboards.append(self.game.game_state_host.bitboards.copy())
        self._food_positions.append(self.game.game_state_host.food_position.copy())
        self._head_positions.append(self.game.game_state_host.head_positions.copy())
        self._moves.append(moves.copy())
        self._move_counts += 1 - self.game.game_state_host.is_dead

    def swap_perspective(self, bitboards):
        swap01 = ((bitboards & 0b00000001) << 1) | ((bitboards & 0b00000010) >> 1)
        swap23 = ((bitboards & 0b00000100) << 1) | ((bitboards & 0b00001000) >> 1)
        result = (bitboards & 0b11110000) | swap01 | swap23
        return result

    def state_to_inputs(self, bitboards, heads, food):
        bitboards = np.kron(bitboards, np.ones((1, 4, 4, 1), dtype=np.uint8))
        shifts = np.tile(
            np.arange(16, dtype=np.uint16).reshape((4, 4)),
            (self.game.size // 4, self.game.size // 4),
        )
        bitboards = bitboards >> shifts[None, ..., None] & 0b1
        shifts = np.arange(2).reshape(1, 1, 1, 2)
        bitboards[range(bitboards.shape[0]), heads[:, 1, 0], heads[:, 0, 0], 0] = 0b0100
        bitboards[range(bitboards.shape[0]), heads[:, 1, 1], heads[:, 0, 1], 1] = 0b0100
        bitboards = (bitboards << shifts).sum(axis=-1, dtype=np.uint8)
        bitboards[range(bitboards.shape[0]), food[:, 1], food[:, 0]] = 0b10000
        return bitboards, self.swap_perspective(bitboards)

    def get_training_data(self, top_k: int):
        self.game.sync_to_host()
        many_body_lenghts = self.game.game_state_host.body_lengths
        many_bitboards = np.array(self._bitboards, dtype=np.uint16)
        many_food_positions = np.array(self._food_positions, dtype=np.uint8)
        many_head_positions = np.array(self._head_positions, dtype=np.uint8)
        many_moves = np.array(self._moves, dtype=np.uint8)

        games = []

        for game in tqdm(range(many_bitboards.shape[1]), desc="Expanding game states"):
            scores = many_body_lenghts[game]
            move_count = self._move_counts[game]
            max_moves = np.max(move_count)
            bitboards = many_bitboards[:max_moves, game]
            heads = many_head_positions[:max_moves, game]
            food = many_food_positions[:max_moves, game]
            moves = many_moves[:max_moves, game]
            bitboards[move_count[0] :, ..., 0] = 0
            bitboards[move_count[1] :, ..., 1] = 0
            player_0_x, player_1_x = self.state_to_inputs(bitboards, heads, food)
            player_0_x = player_0_x[: move_count[0]]
            player_1_x = player_1_x[: move_count[1]]
            player_0_y = moves[: move_count[0], 0]
            player_1_y = moves[: move_count[1], 1]
            games.append((scores[0], player_0_x, player_0_y))
            games.append((scores[1], player_1_x, player_1_y))

        games.sort(key=lambda g: g[0], reverse=True)
        games = games[:top_k]

        batch_x = []
        batch_y = []

        for _, x, y in games:
            batch_x.append(x)
            batch_y.append(y)

        batch_x = np.concatenate(batch_x)
        batch_y = np.concatenate(batch_y)

        return batch_x, batch_y


class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        bs, s, s = x.shape
        x = jnp.reshape(x, (bs, s * s))
        x = jnp.log2(x)
        x = jax.nn.one_hot(x, 6, dtype=jnp.float32)
        x = jnp.reshape(x, (bs, -1))
        x = nn.Dense(512)(x)
        x = nn.silu(x)
        x = nn.Dense(256)(x)
        x = nn.silu(x)
        x = nn.Dense(256)(x)
        x = nn.silu(x)
        x = nn.Dense(128)(x)
        x = nn.silu(x)
        x = nn.Dense(4)(x)
        x = nn.softmax(x)
        return x


def batchify(data_x, data_y, batch_size, rng_key=None):
    num_samples = data_x.shape[0]

    # Shuffle data (optional)
    if rng_key is not None:
        indices = jax.random.permutation(rng_key, num_samples)
        data_x = data_x[indices]
        data_y = data_y[indices]
    else:
        indices = jnp.arange(num_samples)

    # Calculate the number of full batches
    num_full_batches = num_samples // batch_size

    # Function to extract a batch given its index
    def get_batch(batch_idx):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        return (
            data_x[start_idx:end_idx],
            data_y[start_idx:end_idx],
        )

    # Use a list comprehension to create batches
    batches = [get_batch(i) for i in range(num_full_batches)]

    # Handle the last batch if it has fewer elements than batch_size
    if num_samples % batch_size != 0:
        start_idx = num_full_batches * batch_size
        batches.append((data_x[start_idx:], data_y[start_idx:]))

    return batches


if __name__ == "__main__":
    import cv2 as cv
    import time
    import numpy as np
    import cupy as cp

    GAME_SIZE = 8
    NUM_GAMES = 8192
    RENDER_GAMES = 4096
    TOP_K = 64
    MAX_UPDATES = 64
    BATCH_SIZE = 32

    game = CUDASnakeGame(GAME_SIZE, NUM_GAMES)
    game.prepare_cuda()

    game_recorder = GameRecorder(game)

    model = Model()
    key = jax.random.key(0)
    dummy_input = jnp.empty((1, GAME_SIZE, GAME_SIZE), dtype=jnp.uint8)
    variables = model.init(key, dummy_input)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.adam(0.01),
    )

    def loss_fn(params, x, y):
        logits = state.apply_fn({"params": params}, x)
        labels = jax.nn.one_hot(y, 4)
        return -jnp.mean(jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    while True:
        game.reset_games()
        game.sync_to_device()
        game_recorder.initialize_game_recording()

        updates = 0

        pbar = tqdm(total=MAX_UPDATES)

        while games_running := NUM_GAMES - np.sum(
            np.all(game.game_state_host.is_dead, axis=-1)
        ):
            st = time.monotonic()

            if RENDER_GAMES:
                image = game.get_image_from_device(RENDER_GAMES)
                image = np.kron(image, np.ones((4, 4, 1), dtype=np.uint8))
                cv.imshow("image", image)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    quit()

            rng_moves = cp.random.randint(0, 4, (NUM_GAMES, 2), dtype=np.uint8)

            game.sync_to_host()
            in1, in2 = game_recorder.state_to_inputs(
                game.game_state_host.bitboards,
                game.game_state_host.head_positions,
                game.game_state_host.food_position,
            )

            inputs = np.concatenate([in1, in2])
            probs = state.apply_fn({"params": state.params}, inputs)
            moves = jax.random.categorical(
                jax.random.key(np.random.randint(0, int(1e13))), probs
            )
            moves = moves.reshape((2, -1)).T
            moves = cp.from_dlpack(jax.dlpack.to_dlpack(moves))
            # moves[-NUM_GAMES // 4 :, 0] = rng_moves[-NUM_GAMES // 4 :, 0]
            # moves[-NUM_GAMES // 2 :, 1] = rng_moves[-NUM_GAMES // 2 :, 1]

            game_recorder.record_game_step(cp.asnumpy(moves))
            game.increment_games_device(moves)

            frame_time = (time.monotonic() - st) * 1e3
            pbar.set_description(
                f"Games running: {games_running}, Frame time: {frame_time:.5f}ms"
            )
            updates += 1
            pbar.update(1)
            if updates > MAX_UPDATES:
                pbar.set_description(
                    f"Games finished: {NUM_GAMES - games_running}/{NUM_GAMES}"
                )
                break

        pbar.close()
        # cv.destroyAllWindows()

        x, y = game_recorder.get_training_data(TOP_K)

        batches = batchify(x, y, BATCH_SIZE, key)
        for i in range(8):
            for bx, by in tqdm(batches):
                val, grads = grad_fn(state.params, bx, by)
                print(val)
                state = state.apply_gradients(grads=grads)

        with open("params.pickle", "wb") as f:
            pickle.dump(state.params, f)
