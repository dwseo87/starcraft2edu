from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app, flags
from racepack.Terran import TerranBasicAgent
from racepack.Zerg import ZergBasicAgent
from racepack.Protoss import ProtossBasicAgent



def main(unused_argv):
    print(unused_argv)
    race_selected = unused_argv[1]
    if race_selected in ['terran', 't']:
        agent = TerranBasicAgent()
        player = sc2_env.Agent(sc2_env.Race.terran)
    elif race_selected in ['zerg', 'z']:
        agent = ZergBasicAgent()
        player = sc2_env.Agent(sc2_env.Race.zerg)
    elif race_selected in ['protoss', 'p']:
        agent = ProtossBasicAgent()
        player = sc2_env.Agent(sc2_env.Race.protoss)

    try:
        while True: # 권장사항: APM은 very easy 기준 80정도, step mul 20~30. game step은 40000정도로 제한 권장
            with sc2_env.SC2Env(
                    #map_name="AbyssalReef",
                    map_name="Simple64",
                    players=[player,
                             sc2_env.Bot(sc2_env.Race.random,
                                         sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(
                      feature_dimensions=features.Dimensions(screen=84, minimap=64),
                      use_feature_units=True),
                    step_mul=8, #22.4*60/setp_mul 만큼 APM을 설정
                    game_steps_per_episode=0, # 보통 1시간 반정도 게임을 하면 40000step을 넘게 됨. 하나의 게임을 몇번 스텝까지 수행할지 강제할 수 있음
                    visualize=True) as env:

              agent.setup(env.observation_spec(), env.action_spec())

              timesteps = env.reset()
              agent.reset()

              while True:
                  step_actions = [agent.step(timesteps[0])]
                  if timesteps[0].last():
                      break
                  timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--race', help='select race between terran(t), zerg(z), protoss(p)',
                        choices=['terran','zerg','protoss','t','z','p'], default = 'p')

    args = parser.parse_args()
    import sys
    sys.argv.append(args.race)
    app.run(main)