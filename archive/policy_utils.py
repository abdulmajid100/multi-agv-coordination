# policy_utils.py
def get_actions_from_policy(env, agents):
    state = env.reset()
    actions_sequence = []
    done = [False] * env.num_agents  # Track if each agent is done

    while not done:
        actions = [agent.choose_action(tuple(pos)) for agent, pos in zip(agents, state)]
        actions_sequence.append(actions)
        state, _, done, _ = env.step(actions)

    return actions_sequence
