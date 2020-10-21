namespace ML.NET.FSharp

module CarPricePrediction =

    open System
    open Microsoft.ML
    open Microsoft.ML.Data

    [<CLIMutable>]
    type CarInfo = {
        [<LoadColumn(0)>]  [<ColumnName("Label")>] Price : float32
        [<LoadColumn(1)>] Year : float
        [<LoadColumn(2)>] Mileage : float
        [<LoadColumn(6)>] Make : string
        [<LoadColumn(7)>] Model : string
    }

    [<CLIMutable>]
    type CarPriceFarePrediction = {
        [<ColumnName("Score")>] Price : float32
    }

    let dataPath = Common.setPath "car_listings.csv"

    // create the machine learning context
    let context = MLContext(seed = Nullable 1)

    // load the data
    Common.printCyan "Load the data..."
    let dataView = context.Data.LoadFromTextFile<CarInfo>(dataPath, hasHeader = true, separatorChar = ',')

    // split into a training and test partition
    let partitions = context.Data.TrainTestSplit(dataView, testFraction = 0.2)
    let trainSet = partitions.TrainSet
    let testSet = partitions.TestSet

    // set up a learning pipeline
    let pipeline =
        // build a training pipeline
        Common.printCyan "Create pipeline..."

        EstimatorChain()
            // one-hot encode all text features
            .Append(context.Transforms.Categorical.OneHotEncoding("Make"))
            .Append(context.Transforms.Categorical.OneHotEncoding("Model"))
            .Append(context.Transforms.Conversion.ConvertType("Year", outputKind = DataKind.Single))
            .Append(context.Transforms.Conversion.ConvertType("Mileage", outputKind = DataKind.Single))

            // combine all input features into a single column
            .Append(context.Transforms.Concatenate("Features", "Year", "Mileage", "Make", "Model"))

            // cache the data to speed up training
            .AppendCacheCheckpoint(context)
            .Append(context.Regression.Trainers.Ols())
            //.Append(context.Regression.Trainers.FastTree(numberOfTrees = 200, minimumExampleCountPerLeaf = 4))
            // .Append(context.Regression.Trainers.FastForest(numberOfTrees = 200, minimumExampleCountPerLeaf = 4))

    // train the model
    Common.printCyan "Training the model..."
    let model = trainSet |> pipeline.Fit

    // get regression metrics to score the model
    Common.printCyan "Training the model..."

    let metrics = testSet |> model.Transform |> context.Regression.Evaluate
    // show the metrics
    Common.printGreen (sprintf "Model metrics:")
    Common.printGreen (sprintf "  R2 score: %A" metrics.RSquared)
    Common.printGreen (sprintf "  RMSE:%f" metrics.RootMeanSquaredError)
    Common.printGreen (sprintf "  MSE: %f" metrics.MeanSquaredError)
    Common.printGreen (sprintf "  MAE: %f" metrics.MeanAbsoluteError)


    // Evaluate the model again using cross-validation
    Common.printRed "Cross validation..."
    let scores = context.Regression.CrossValidate(data = dataView, estimator = Common.castToEstimator pipeline, numberOfFolds = 5)
    let mean = scores |> Seq.averageBy(fun x -> x.Metrics.RSquared)
    Common.printRed (sprintf "Mean cross-validated R2 score: %A" mean)

    // create a prediction engine for one single prediction
    Common.printCyan "Create Prediction Engine"
    let engine : PredictionEngine<CarInfo, CarPriceFarePrediction> = context.Model.CreatePredictionEngine model

    let runPrediction () =
        let carInfoPorsche = {
            Year = 2018.
            Mileage = 12500.
            Make = "Porsche"
            Model = "MacanAWD"
            Price = 0.0f // To predict.
        }
        let carInfoBMW = {
            Year = 2012.
            Mileage = 26000.
            Make = "BMW"
            Model = "X5AWD"
            Price = 0.0f // To predict.
        }

        // make the prediction
        let predictionBMW = carInfoBMW |> engine.Predict
        let predictionPORSCHE = carInfoPorsche |> engine.Predict

        // show the prediction
        Common.printRed (sprintf "Single prediction:")
        Common.printRed (sprintf "  Predicted Price for %s-%s (%A) : %f" carInfoBMW.Make carInfoBMW.Model carInfoBMW.Year predictionBMW.Price)
        Common.printRed (sprintf "Single prediction:")
        Common.printRed (sprintf "  Predicted Price for %s-%s (%A) : %f" carInfoPorsche.Make carInfoPorsche.Model carInfoPorsche.Year predictionPORSCHE.Price)
