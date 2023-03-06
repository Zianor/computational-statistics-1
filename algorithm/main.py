from algorithm.instantiation import CausalDiscoveryGA

def main():
    causalGA = CausalDiscoveryGA()
    causalGA.initialize_env()
    pop = causalGA.start_ga()
    print(pop)

if __name__ == "__main__":
    main()
