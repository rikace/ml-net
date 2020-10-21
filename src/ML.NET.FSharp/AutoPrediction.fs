namespace ML.NET.FSharp

module AutoPrediction =

    open System
    open Microsoft.ML
    open Microsoft.ML.AutoML
    open Microsoft.ML.Data

    [<CLIMutable>]
    type CarInfo = {
        [<LoadColumn(0)>]  [<ColumnName("Label")>] Price : float32
        [<LoadColumn(1)>] Year : string
        [<LoadColumn(2)>] Mileage : string
        [<LoadColumn(6)>] Make : string
        [<LoadColumn(7)>] Model : string
    }

    [<CLIMutable>]
    type CarPriceFarePrediction = {
        [<ColumnName("Score")>] Price : float32
    }

    let dataPath = Common.setPath "car_listings.csv"

    // create the machine learning context
    let context = new MLContext(seed = Nullable 1)

    // load the data
    Common.print "Load Data..."
    let dataView = context.Data.LoadFromTextFile<CarInfo>(dataPath, hasHeader = true, separatorChar = ',')

    // Create an experiment
    let settings = RegressionExperimentSettings
                        (
                            MaxExperimentTimeInSeconds = 30u,
                            OptimizingMetric = RegressionMetric.RSquared,
                            CacheDirectory = null
                        )

    // Run the experiment
    Common.printCyan "Create Auto..."
    let experiment = context.Auto().CreateRegressionExperiment(settings)

    Common.printCyan "Running the experiment..."
    let result = experiment.Execute(dataView);

    Common.printYellow "Run Validation Metrics..."
    let metrics = result.BestRun.ValidationMetrics
    Common.printGreen (sprintf "Model metrics:")
    Common.printGreen (sprintf "  R2 score: %A" metrics.RSquared)
    Common.printGreen (sprintf "  RMSE:%f" metrics.RootMeanSquaredError)
    Common.printGreen (sprintf "  MSE: %f" metrics.MeanSquaredError)
    Common.printGreen (sprintf "  MAE: %f" metrics.MeanAbsoluteError)

    // create a prediction engine for one single prediction
    let engine : PredictionEngine<CarInfo, CarPriceFarePrediction> = context.Model.CreatePredictionEngine result.BestRun.Model

    let runPrediction () =
        let carInfo = {
            Year = "2018"
            Mileage = "12500"
            Make = "Porsche"
            Model = "MacanAWD"
            Price = 0.0f // To predict.=
        }

        Common.printCyan (sprintf "Test Prediction with data: %A" carInfo)

        // make the prediction
        let prediction = carInfo |> engine.Predict

        // show the prediction
        Common.printGreen (sprintf "\r")
        Common.printGreen (sprintf "Single prediction:")
        Common.printGreen (sprintf "  Predicted fare: %f" prediction.Price)

