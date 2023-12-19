
import numpy as np
import datetime as dt
import pyswarm
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pygad
import matplotlib.pyplot as plt

# Tkintermodern gui
import TKinterModernThemes as TKMT
import tkinter as tk
from PIL import ImageTk, Image

class App(TKMT.ThemedTKinterFrame):
    def __init__(self, theme, mode, usecommandlineargs=True, usethemeconfigfile=True):
        super().__init__("Portfolio Optimization", theme, mode,
                         usecommandlineargs=usecommandlineargs, useconfigfile=usethemeconfigfile)

        self.text_input_var = tk.StringVar(value="Enter Desired Return")

        # Create a Frame for input widgets
        self.input_frame = self.addLabelFrame("Return Expectations", col=0, colspan=1)
        self.input_frame.Entry(self.text_input_var)
        self.input_frame.Button("Process", self.handle_submit_button)

        # Create tabs
        self.notebook = self.Notebook("Tabs", col=1, colspan=3)
        self.tabs = {}  # Dictionary to store references to tabs

        tab1 = self.notebook.addTab("Particle Swarm")
        tab1.Text("PSO", row=0)
        self.tabs["Particle Swarm"] = tab1

        tab2 = self.notebook.addTab("Genetic Algo")
        tab2.Text("Genetic",row=0)
        self.tabs["Genetic Algo"] = tab2

        tab3 = self.notebook.addTab("Simulated Annealing")
        tab3.Text("Simulated Annealing",row=0)
        self.tabs["Simulated Annealing"] = tab3

        tab4 = self.notebook.addTab("Hill Climb Optimization")
        tab4.Text("Hill Climb")
        self.tabs["Hill Climb Optimization"] = tab4

        tab5 = self.notebook.addTab("Analysis")
        tab5.Text("Analysis")
        self.tabs["Analysis"] = tab5

        self.run()

    def handle_submit_button(self):
        text_value = self.text_input_var.get()
        converted = float(text_value)
        psoreturn, psorisk, simalreturn, simalrisk, hillclimbreturn, hillclimbrisk, geneticreturn, geneticrisk = runfn(
            converted)

        # Update text for each tab with respective portfolio statistics
        self.tabs["Particle Swarm"].Text(f'Portfolio Return: {psoreturn: .3f}\nPortfolio Risk: {psorisk: .3f}', fontargs=("Helvetica", 16, "bold"))
        self.tabs["Genetic Algo"].Text(f'Portfolio Return: {geneticreturn: .3f}\nPortfolio Risk: {geneticrisk: .3f}', fontargs=("Helvetica", 16, "bold"))
        self.tabs["Simulated Annealing"].Text(f'Portfolio Return: {simalreturn: .3f}\nPortfolio Risk: {simalrisk: .3f}', fontargs=("Helvetica", 16, "bold"))
        self.tabs["Hill Climb Optimization"].Text(
            f'Portfolio Return: {hillclimbreturn: .3f}\nPortfolio Risk: {hillclimbrisk: .3f}', fontargs=("Helvetica", 16, "bold"))

        # Load and display images for each search method
        self.load_and_display_image("pso_output.jpg", "Particle Swarm")
        self.load_and_display_image("genetic_output.jpg", "Genetic Algo")
        self.load_and_display_image("sa_output.jpg", "Simulated Annealing")
        self.load_and_display_image("hillClimb_output.jpg", "Hill Climb Optimization")

        # Create bar graph
        labels = ['Particle Swarm', 'Genetic Algo', 'Simulated Annealing', 'Hill Climb Optimization']
        returns = [psoreturn, geneticreturn, simalreturn, hillclimbreturn]
        risks = [psorisk, geneticrisk, simalrisk, hillclimbrisk]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # Create 1 row, 2 columns of subplots

        # Bar graph on first subplot
        rects1 = axs[0].bar(x - width / 2, returns, width, label='Return')
        rects2 = axs[0].bar(x + width / 2, risks, width, label='Risk')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        axs[0].set_ylabel('Scores')
        axs[0].set_title('Scores by algorithm and metric')
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(labels)
        axs[0].legend()

        # Calculate Sharpe ratios
        risk_free_rate = 0.02  # Assuming a risk-free rate of 2%
        pso_sharpe = (psoreturn - risk_free_rate) / psorisk
        genetic_sharpe = (geneticreturn - risk_free_rate) / geneticrisk
        simal_sharpe = (simalreturn - risk_free_rate) / simalrisk
        hillclimb_sharpe = (hillclimbreturn - risk_free_rate) / hillclimbrisk

        # Line chart for Sharpe ratios on second subplot
        sharpe_ratios = [pso_sharpe, genetic_sharpe, simal_sharpe, hillclimb_sharpe]
        axs[1].plot(labels, sharpe_ratios, marker='o')
        axs[1].set_title('Sharpe Ratios by Algorithm')
        axs[1].set_xlabel('Algorithm')
        axs[1].set_ylabel('Sharpe Ratio')

        fig.tight_layout()

        # Save the figure
        plt.savefig("analysis_output.jpg")

        # Load and display the image in the "Analysis" tab
        self.load_and_display_image("analysis_output.jpg", "Analysis", (1000,1000))

    def load_and_display_image(self, image_path, tab_name , size=(700,700)):
        # Load the image using PIL
        image = Image.open(image_path)
        image.thumbnail(size)
        image = ImageTk.PhotoImage(image)

        # Create a Canvas widget and add it to the respective tab using grid
        canvas = tk.Canvas(self.tabs[tab_name].master)  # Pass the master widget explicitly
        canvas.grid(row=0, column=0, sticky=tk.NSEW)  # Adjust row and column as needed
        canvas.create_image(0, 0, anchor=tk.NW, image=image)

        # Keep a reference to the image to prevent it from being garbage collected
        canvas.image = image


def pre_process(num):
    global MONTHS_IN_YEAR, minDesiredReturn, StockTickers, ExpectedReturns, VarCov
    maxDate = dt.datetime(2022, 12, 31)  # Starting date is set to January 1, 2012
    minDate = dt.datetime(2012, 1, 1)  # Ending date set to December 21, 2022
    # Define Constants and Parameters
    MONTHS_IN_YEAR = 12
    minDesiredReturn = num
    
    from StockData.StockPull import pullStockData

    # List of stock tickers to pull data from (top 50 market cap stocks companies in US S&P 500)
    StockTickers = ['AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'TSLA', 'JPM']
    RawStockData = pullStockData(StockTickers, minDate, maxDate)
    # Data processing and calculations
    # Calculate monthly returns
    RawStockData['adjClose'] = RawStockData['adjClose'].astype(float)  # Ensure data type is float
    RawStockData = RawStockData.sort_values(by=['stock', 'period'])
    Returns = RawStockData.pivot(index='period', columns='stock', values='adjClose').pct_change().dropna()
    # Calculate Expected Returns
    ExpectedReturns = Returns.mean()
    # Calculate Excess Returns
    ExcessReturns = Returns - ExpectedReturns
    # Calculate Variance-Covariance Matrix
    numPeriods = len(ExcessReturns.index)
    VarCov = ExcessReturns.cov()


# Define a function to calculate portfolio statistics
def calculate_portfolio_statistics(weights, ExpectedReturns, VarCov):
    portfolio_return = np.dot(weights, ExpectedReturns) * MONTHS_IN_YEAR
    portfolio_variance = np.dot(weights, np.dot(VarCov, weights)) * MONTHS_IN_YEAR
    portfolio_risk = np.sqrt(portfolio_variance)
    return portfolio_return, portfolio_risk


def evaluate_portfolio(weights, ExpectedReturns, VarCov, minDesiredReturn):
    portfolio_return, portfolio_risk = calculate_portfolio_statistics(weights, ExpectedReturns, VarCov)

    if portfolio_return < minDesiredReturn:
        # Return a large value to exclude this portfolio from consideration
        return float('inf')

    # Constrain the sum of weights to be equal to 1
    weight_sum_constraint_penalty = abs(weights.sum() - 1)  # You can adjust the penalty as needed

    return portfolio_risk + weight_sum_constraint_penalty


def hill_Climbing():
    num_iterations = 10000

    # Function to calculate objective value (portfolio risk)
    def objective_function_risk(weights, VarCov):
        portfolio_risk = np.dot(weights, np.dot(VarCov, weights)) * MONTHS_IN_YEAR
        return portfolio_risk

    # Hill climb search algorithm for risk minimization
    def hill_climb_search_risk(ExpectedReturns, VarCov, minDesiredReturn):
        current_weights = np.ones(len(StockTickers)) / len(StockTickers)
        current_risk = objective_function_risk(current_weights, VarCov)

        for _ in range(num_iterations):
            # Generate a neighboring solution by perturbing the current weights
            neighbor_weights = current_weights + np.random.normal(0, 0.05, len(StockTickers))
            neighbor_weights = np.clip(neighbor_weights, 0, 1)
            neighbor_weights /= neighbor_weights.sum()

            # Evaluate the objective function for the neighbor
            neighbor_risk = objective_function_risk(neighbor_weights, VarCov)
            neighbor_return = np.dot(neighbor_weights, ExpectedReturns)

            # If the neighbor is better within the desired return range, update the current solution
            if (minDesiredReturn - 1) <= neighbor_return <= (minDesiredReturn + 0.5) and neighbor_risk < current_risk:
                current_weights = neighbor_weights
                current_risk = neighbor_risk

        return current_weights

    # Run the hill climb search algorithm for risk minimization
    best_weights = hill_climb_search_risk(ExpectedReturns, VarCov, minDesiredReturn)

    # Calculate and print the best portfolio's statistics
    best_portfolio_return, best_portfolio_risk = calculate_portfolio_statistics(best_weights, ExpectedReturns, VarCov)

    print("Portfolio Weights:", best_weights)
    print("Portfolio Return:", best_portfolio_return)
    print("Portfolio Risk:", best_portfolio_risk)

    fig = px.pie(
        names=StockTickers,
        values=best_weights,
        title="Optimized Portfolio with Weights",
    )

    # Increase the dimensions of the pie chart by adjusting the width and height
    fig.update_layout(
        width=800,  # Set the width to the desired value (in pixels)
        height=800  # Set the height to the desired value (in pixels)
    )

    fig.update_traces(textinfo='percent+label')  # Adjust pull for labels
    pio.write_image(fig, 'hillClimb_output.jpg')
    return best_portfolio_return, best_portfolio_risk


def simulated_annealing():
    num_iterations = 10000

    # Function to calculate objective value (portfolio risk)
    def objective_function_risk(weights, VarCov):
        portfolio_risk = np.dot(weights, np.dot(VarCov, weights)) * MONTHS_IN_YEAR
        return portfolio_risk

    # Simulated Annealing algorithm for risk minimization
    def simulated_annealing_risk(ExpectedReturns, VarCov, minDesiredReturn, initial_temperature=9000, cooling_rate=0.95,
                                 iterations_per_temperature=100):
        current_weights = np.ones(len(StockTickers)) / len(StockTickers)
        current_risk = objective_function_risk(current_weights, VarCov)

        best_weights = current_weights
        best_risk = current_risk

        temperature = initial_temperature


        for _ in range(iterations_per_temperature):
            # Generate a neighboring solution by perturbing the current weights
            neighbor_weights = current_weights + np.random.normal(0, 0.05, len(StockTickers))
            neighbor_weights = np.clip(neighbor_weights, 0, 1)
            neighbor_weights /= neighbor_weights.sum()

            # Evaluate the objective function for the neighbor
            neighbor_risk = objective_function_risk(neighbor_weights, VarCov)

            # If the neighbor is better or accepted with a certain probability, update the current solution
            if neighbor_risk < current_risk or np.random.rand() < np.exp((current_risk - neighbor_risk) / temperature):
                current_weights = neighbor_weights
                current_risk = neighbor_risk

                curr_return = np.dot(current_weights, ExpectedReturns)
                if (minDesiredReturn - 1) <= curr_return <= (minDesiredReturn + 0.5) and current_risk < best_risk:
                    best_weights = current_weights
                    best_risk = current_risk

            # Cool down the temperature
            temperature *= cooling_rate

        return best_weights

    best_weights = simulated_annealing_risk(ExpectedReturns, VarCov, minDesiredReturn)

    # Calculate and print the best portfolio's statistics
    best_portfolio_return, best_portfolio_risk = calculate_portfolio_statistics(best_weights, ExpectedReturns, VarCov)

    print("Portfolio Weights:", best_weights)
    print("Portfolio Return:", best_portfolio_return)
    print("Portfolio Risk:", best_portfolio_risk)

    fig = px.pie(
        names=StockTickers,
        values=best_weights,
        title="Optimized Portfolio with Weights",
    )

    # Increase the dimensions of the pie chart by adjusting the width and height
    fig.update_layout(
        width=800,  # Set the width to the desired value (in pixels)
        height=800  # Set the height to the desired value (in pixels)
    )

    fig.update_traces(textinfo='percent+label')  # Adjust pull for labels
    pio.write_image(fig, 'sa_output.jpg')
    return best_portfolio_return, best_portfolio_risk


def PSO_Search():
    max_iterations = 10
    numPortfolios = 50
    lowerBound = [0] * len(StockTickers)
    upperBound = [1] * len(StockTickers)
    best_solutions = []
    best_solutions = []
    for _ in range(max_iterations):
        weights, _ = pyswarm.pso(
            evaluate_portfolio, lowerBound, upperBound,
            args=(ExpectedReturns, VarCov, minDesiredReturn),
            swarmsize=numPortfolios, maxiter=1000, debug=False
        )

        best_solutions.append(weights)
    best_weights = min(best_solutions, key=lambda x: evaluate_portfolio(x, ExpectedReturns, VarCov, minDesiredReturn))
    # Calculate and print the best portfolio's statistics
    best_portfolio_return, best_portfolio_risk = calculate_portfolio_statistics(best_weights, ExpectedReturns, VarCov)
    print("Portfolio Weights:", best_weights)
    print("Portfolio Return:", best_portfolio_return)
    print("Portfolio Risk:", best_portfolio_risk)
    fig = px.pie(
        names=StockTickers,
        values=best_weights,
        title="Optimized Portfolio with Weights",
    )
    # Increase the dimensions of the pie chart by adjusting the width and height
    fig.update_layout(
        width=800,  # Set the width to the desired value (in pixels)
        height=800  # Set the height to the desired value (in pixels)
    )
    fig.update_traces(textinfo='percent+label')  # Adjust pull for labels
    pio.write_image(fig, 'pso_output.jpg')
    return best_portfolio_return, best_portfolio_risk


def genetic_algorithm():
    # Define Genetic Algorithm parameters
    num_generations = 300
    num_parents_mating = 30
    population_size = 300
    mutation_percent_genes = 20

    # Define the bounds for portfolio weights (between 0 and 1)
    num_assets = len(StockTickers)
    initial_population = []
    for _ in range(population_size):
        weights = np.random.rand(num_assets)
        weights /= np.sum(weights)
        initial_population.append(weights)

    def fitness_func(solution, solution_idx):
        if np.all(0 <= solution) and np.all(solution <= 1) and np.isclose(np.sum(solution), 1.0, rtol=1e-5):
            return 1.0 / evaluate_portfolio(solution, ExpectedReturns, VarCov, minDesiredReturn)
        else:
            return 0.0

    # Create a PyGAD object
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           initial_population=initial_population,
                           fitness_func=fitness_func,
                           num_genes=num_assets,
                           mutation_percent_genes=mutation_percent_genes)

    ga_instance.run()

    # Get the best solution (portfolio weights)
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    best_portfolio_return, best_portfolio_risk = calculate_portfolio_statistics(best_solution, ExpectedReturns, VarCov)

    print("Best Portfolio Weights:", best_solution)
    print("Best Portfolio Return:", best_portfolio_return)
    print("Best Portfolio Risk:", best_portfolio_risk)

    fig = px.pie(
        names=StockTickers,
        values=best_solution,
        title="Optimized Portfolio with Weights",
    )

    # Increase the dimensions of the pie chart by adjusting the width and height
    fig.update_layout(
        width=800,  # Set the width to the desired value (in pixels)
        height=800  # Set the height to the desired value (in pixels)
    )

    fig.update_traces(textinfo='percent+label')  # Adjust pull for labels
    pio.write_image(fig, 'genetic_output.jpg')
    return best_portfolio_return, best_portfolio_risk


def runfn(num):
    pre_process(num)
    psoreturn, psorisk = PSO_Search()
    simalreturn, simalrisk = simulated_annealing()
    hillclimbreturn, hillclimbrisk = hill_Climbing()
    geneticreturn, geneticrisk = genetic_algorithm()
    return psoreturn, psorisk, simalreturn ,simalrisk, hillclimbreturn, hillclimbrisk, geneticreturn, geneticrisk


if __name__ == "__main__":
    app = App("azure", "dark")