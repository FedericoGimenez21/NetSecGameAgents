# Authors:  Ondrej Lukas - ondrej.lukas@aic.fel.cvut.cz
#           Arti
#           Sebastian Garcia. sebastian.garcia@agents.fel.cvut.cz
# Version sin logging para uso en GA optimization
import sys
import numpy as np
import random
import pickle
import argparse
import time

from os import path
# with the path fixed, we can import now
from AIDojoCoordinator.game_components import Action, Observation, GameState, AgentStatus
from NetSecGameAgents.agents.base_agent import BaseAgent
from NetSecGameAgents.agents.agent_utils import generate_valid_actions, state_as_ordered_string
from feature_extractor import FeatureExtractor

class QAgent(BaseAgent):

    def __init__(self, host, port, role="Attacker", alpha=0.1, gamma=0.6, epsilon_start=0.9, epsilon_end=0.1, epsilon_max_episodes=5000, apm_limit:int=None) -> None:
        super().__init__(host, port, role)
        self.alpha = alpha
        self.gamma = gamma
        self.q_values = {}
        self._str_to_id = {}
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_max_episodes = epsilon_max_episodes
        self.current_epsilon = epsilon_start
        self._apm_limit = apm_limit
        # Simplificar la inicialización del extractor de características
        self.feature_extractor = FeatureExtractor()
        if self._apm_limit:
            self.inter_action_interval = 60/apm_limit
        else:
            self.inter_action_interval = 0

    def store_q_table(self, filename):
        """Simplificar el almacenamiento"""
        with open(filename, "wb") as f:
            data = {
                "q_table": self.q_values,
                "state_mapping": self._str_to_id
            }
            pickle.dump(data, f)
    
    def load_q_table(self,filename):
        """Simplificar la carga"""
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.q_values = data["q_table"]
                self._str_to_id = data["state_mapping"]
        except Exception as e:
            print(f'Error loading file {filename}. {e}')
            sys.exit(-1)

    def get_state_id(self, state:GameState) -> tuple:
        """
        Extrae características del estado y las usa como identificador
        """
        state_str = state_as_ordered_string(state)
        features = self.feature_extractor.extract_features(state_str)
        # Discretizar características para usar como clave en Q-table
        return tuple(int(x) for x in features)

    def max_action_q(self, observation:Observation) -> Action:
        state = observation.state
        actions = generate_valid_actions(state)
        state_id = self.get_state_id(state)
        tmp = dict(((state_id, a), self.q_values.get((state_id, a), 0)) for a in actions)
        return tmp[max(tmp,key=tmp.get)] #return maximum Q_value for a given state (out of available actions)
   
    def select_action(self, observation:Observation, testing=False) -> tuple:
        state = observation.state
        actions = generate_valid_actions(state)
        state_id = self.get_state_id(state)

        # E-greedy play. If the random number is less than the e, then choose random to explore.
        # But do not do it if we are testing a model. 
        if random.uniform(0, 1) <= self.current_epsilon and not testing:
            # We are training
            # Random choose an ation from the list of actions?
            action = random.choice(list(actions))
            if (state_id, action) not in self.q_values:
                self.q_values[state_id, action] = 0
            return action, state_id
        else: 
            # Here we can be during training outside the e-greede, or during testing
            # Select the action with highest q_value, or random pick to break the ties
            # The default initial q-value for a (state, action) pair is 0.
            initial_q_value = 0
            tmp = dict(((state_id, action), self.q_values.get((state_id, action), initial_q_value)) for action in actions)
            ((state_id, action), value) = max(tmp.items(), key=lambda x: (x[1], random.random()))
            #if max_q_key not in self.q_values:
            try:
                self.q_values[state_id, action]
            except KeyError:
                self.q_values[state_id, action] = 0
            return action, state_id

    def recompute_reward(self, observation: Observation) -> Observation:
        """
        Redefine how q-learning recomputes the inner reward
        """
        new_observation = None
        state = observation.state
        reward = observation.reward
        end = observation.end
        info = observation.info

        if info and info['end_reason'] == AgentStatus.Fail:
            reward = -1000
        elif info and info['end_reason'] == AgentStatus.Success:
            reward = 1000
        elif info and info['end_reason'] == AgentStatus.TimeoutReached:
            reward = -100
        else:
            reward = -1
        
        new_observation = Observation(state, reward, end, info)
        return new_observation

    def update_epsilon_with_decay(self, episode_number)->float:
        decay_rate = np.max([(self.epsilon_max_episodes - episode_number) / self.epsilon_max_episodes, 0])
        new_eps = (self.epsilon_start - self.epsilon_end ) * decay_rate + self.epsilon_end
        return new_eps
    
    def play_game(self, observation, episode_num, testing=False):
        """
        The main function for the gameplay. Handles the main interaction loop.
        """
        num_steps = 0
        # Run the whole episode
        while not observation.end:
            # Store steps so far
            num_steps += 1
            start_time = time.time()
            # Get next action. If we are not training, selection is different, so pass it as argument
            action, state_id = self.select_action(observation, testing)
            # Perform the action and observe next observation
            observation = self.make_step(action)
           
            # Recompute the rewards
            observation = self.recompute_reward(observation)
            if not testing:
                # If we are training update the Q-table
                self.q_values[state_id, action] += self.alpha * (observation.reward + self.gamma * self.max_action_q(observation)) - self.q_values[state_id, action]

            # Check the apm (actions per minute)
            if self._apm_limit:
                elapsed_time = time.time() - start_time
                remaining_time = self.inter_action_interval - elapsed_time
                if remaining_time > 0:
                    # We still have some time in this interval, but we can not
                    # take more actions. So wait until the next interval starts
                    time.sleep(remaining_time)
                start_time = time.time()

        # update epsilon value
        if not testing:
            self.current_epsilon = self.update_epsilon_with_decay(episode_num)
        # Reset the episode
        _ = self.request_game_reset()
        # This will be the last observation played before the reset
        return observation, num_steps

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Q-learning agent without logging - optimized for GA')
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    parser.add_argument("--episodes", help="Sets number of episodes to run.", default=15000, type=int)
    parser.add_argument("--epsilon_start", help="Sets the start epsilon for exploration during training.", default=0.9, type=float)
    parser.add_argument("--epsilon_end", help="Sets the end epsilon for exploration during training.", default=0.1, type=float)
    parser.add_argument("--epsilon_max_episodes", help="Max episodes for epsilon to reach maximum decay", default=8000, type=int)
    parser.add_argument("--gamma", help="Sets gamma discount for Q-learing during training.", default=0.9, type=float)
    parser.add_argument("--alpha", help="Sets alpha for learning rate during training.", default=0.1, type=float)
    parser.add_argument("--previous_model", help="Load the previous model. If training, it will start from here. If testing, will use to test.", type=str)
    parser.add_argument("--testing", help="Test the agent. No train.", default=False, type=bool)
    parser.add_argument("--experiment_id", help="Id of the experiment.", default='', type=str)
    parser.add_argument("--apm", help="Actions per minute", default=10000, type=int, required=False)
    args = parser.parse_args()

    # Create agent
    agent = QAgent(args.host, args.port, alpha=args.alpha, gamma=args.gamma, epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end, epsilon_max_episodes=args.epsilon_max_episodes, apm_limit=args.apm)

    # If there is a previous model passed. Always use it for both training and testing.
    if args.previous_model:
        try:
            agent.load_q_table(args.previous_model)
        except FileNotFoundError:
            message = f'Problem loading the file: {args.previous_model}'
            print(message)
            sys.exit(-1)

    # Register the agent
    observation = agent.register()

    try:
        # To keep statistics of each episode
        wins = 0
        detected = 0
        max_steps = 0
        num_win_steps = []
        num_detected_steps = []
        num_max_steps_steps = []
        num_detected_returns = []
        num_win_returns = []
        num_max_steps_returns = []

        for episode in range(1, args.episodes + 1):
            # Play 1 episode
            observation, num_steps = agent.play_game(observation, testing=args.testing, episode_num=episode)       

            state = observation.state
            reward = observation.reward
            end = observation.end
            info = observation.info

            if observation.info and observation.info['end_reason'] == AgentStatus.Fail:
                detected +=1
                num_detected_steps += [num_steps]
                num_detected_returns += [reward]
            elif observation.info and observation.info['end_reason'] == AgentStatus.Success:
                wins += 1
                num_win_steps += [num_steps]
                num_win_returns += [reward]
            elif observation.info and observation.info['end_reason'] == AgentStatus.TimeoutReached:
                max_steps += 1
                num_max_steps_steps += [num_steps]
                num_max_steps_returns += [reward]

            # Reset the game
            observation = agent.request_game_reset()

        # Calcular métricas finales
        eval_win_rate = (wins/args.episodes) * 100
        eval_detection_rate = (detected/args.episodes) * 100
        eval_average_returns = np.mean(num_detected_returns + num_win_returns + num_max_steps_returns)
        eval_std_returns = np.std(num_detected_returns + num_win_returns + num_max_steps_returns)
        eval_average_episode_steps = np.mean(num_win_steps + num_detected_steps + num_max_steps_steps)
        eval_std_episode_steps = np.std(num_win_steps + num_detected_steps + num_max_steps_steps)
        eval_average_win_steps = np.mean(num_win_steps)
        eval_std_win_steps = np.std(num_win_steps)
        eval_average_detected_steps = np.mean(num_detected_steps)
        eval_std_detected_steps = np.std(num_detected_steps)
        eval_average_max_steps_steps = np.mean(num_max_steps_steps)
        eval_std_max_steps_steps = np.std(num_max_steps_steps)
        # Print final stats
        text = f'''Final model performance after {args.episodes} episodes.
                Wins={wins},
                Detections={detected},
                winrate={eval_win_rate:.3f}%,
                detection_rate={eval_detection_rate:.3f}%,
                average_returns={eval_average_returns:.3f} +- {eval_std_returns:.3f},
                average_episode_steps={eval_average_episode_steps:.3f} +- {eval_std_episode_steps:.3f},
                average_win_steps={eval_average_win_steps:.3f} +- {eval_std_win_steps:.3f},
                average_detected_steps={eval_average_detected_steps:.3f} +- {eval_std_detected_steps:.3f}
                average_max_steps_steps={eval_std_max_steps_steps:.3f} +- {eval_std_max_steps_steps:.3f},
                epsilon={agent.current_epsilon}
            '''
        print(text)
        agent.terminate_connection()

    except KeyboardInterrupt:
        # Store the q-table
        if not args.testing:
            agent.store_q_table(f'q_agent_marl.experiment{args.experiment_id}.pickle')
    finally:
        # Store the q-table
        if not args.testing:
            agent.store_q_table(f'q_agent_marl.experiment{args.experiment_id}.pickle')
