import numpy as np
import wandb
from collections import defaultdict 
from tqdm import tqdm 

def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video(renders=None, n_cols=None, fps=15):
    from PIL import Image, ImageEnhance
    """Return a Weights & Biases video.

    It takes a list of videos and reshapes them into a single video with the specified number of columns.

    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # Add borders.
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)
    
    return wandb.Video(renders, fps=fps, format='mp4')


def evaluate_ogbench(agent, env, evals, eval_tasks, t, 
                     eval_freq=50000, eval_eps=20,
                     video_eps=2, skip_wandb=False):
    
    if t == 0 or t % eval_freq != 0:
        return {}
    
    renders = []
    eval_metrics, overall_metrics = {}, defaultdict(list)
    # video_eps = 5
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    num_tasks = eval_tasks if eval_tasks is not None else len(task_infos)
    for task_id in tqdm(range(1, num_tasks + 1)):
        task_name = task_infos[task_id - 1]['task_name']
        eval_info, cur_renders = evaluate_fn(
            agent=agent,
            env=env,
            task_id=task_id,
            args=None,
            num_eval_episodes=eval_eps,
            num_video_episodes=video_eps,
            video_frame_skip=3,
            eval_gaussian=None,
        )
        renders.extend(cur_renders)
        metric_names = ['success']
        eval_metrics.update(
            {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
        )
        for k, v in eval_info.items():
            if k in metric_names:
                overall_metrics[k].append(v)
    for k, v in overall_metrics.items():
        eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)
    
    results = eval_metrics.copy()
    
    if video_eps > 0 and not skip_wandb:
        video = get_wandb_video(renders=renders, n_cols=num_tasks)
        eval_metrics['video'] = video
        wandb.log(eval_metrics, step=t)
    
    
    # print(f'--------------------- {t} ---------------------')
    tqdm.write(f"Score {eval_metrics['evaluation/overall_success']}/ 1")
    # print(f'--------------------- {t} ---------------------')
    evals.append(eval_metrics)
    return results
    

def evaluate_fn(
    agent,
    env,
    task_id=None,
    args=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        args: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    # trajs = []
    stats = defaultdict(list)

    renders = []
    for i in range(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes
        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []
        while not done:
            # action = agent(observations=observation, goals=goal, temperature=eval_temperature)
            action = agent.select_action(observation, goal, temperature=0.0001)
            action = np.array(action)
 
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            # trajs.append(traj)
        else:
            add_to(stats, flatten(info))
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, renders