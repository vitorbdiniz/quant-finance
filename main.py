import source.portfolio.prices as prices
import source.portfolio.benchmark as benchmark

def main():

    #benchmark.get_benchmark(benchmark='IBXX', update=True)
    indexes = benchmark.all_compositions(False)
    

if __name__ == "__main__":
    main()