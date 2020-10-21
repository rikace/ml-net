namespace ML.NET.FSharp

module DemoCarPricePrediction =

    open System
    open Microsoft.ML
    open Microsoft.ML.Data

    [<CLIMutable>]
    type CarInfo = {
        [<LoadColumn(0)>] [<ColumnName("Label")>] Price : float32
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

    // TODO
    // Step 1 - Create MLContext
    // Step 2 - Load the data
    // Step 3 - Split the Data
    // Step 4 - Convert the Data /  Create Pipeline (define workflow)
    // Step 5 - Training the algorithm for the model
    // Step 6 - Use the test data against the model
    // Step 7 - Check accuracy and improve (cross-validation)
    // Step 8 - Use the model



    // Step 1 - Create MLContext

    // Step 2 - Load the data
    Common.printCyan "Load the data..."


    // Step 3 - Split the Data


    // Step 4 - Convert the Data /  Create Pipeline (define workflow)
    let pipeline =
        Common.printCyan "Create pipeline..."


    // Step 5 - Training the algorithm for the model
    Common.printCyan "Training the model..."


    // Step 6 - Use the test data against the model


    // show the metrics
    Common.printGreen (sprintf "Model metrics:")
//    Common.printGreen (sprintf "  RMSE:%f" metrics.RootMeanSquaredError)
//    Common.printGreen (sprintf "  R2 score: %A" metrics.RSquared)
//    Common.printGreen (sprintf "  MSE: %f" metrics.MeanSquaredError)
//    Common.printGreen (sprintf "  MAE: %f" metrics.MeanAbsoluteError)

    // Step 7 - Check accuracy and improve (cross-validation)
    Common.printRed "Cross validation..."



    //Common.printRed (sprintf "Mean cross-validated R2 score: %A" mean)

    // Step 8 - Use the model
    Common.printCyan "Create Prediction Engine"


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
        let predictionBMW = carInfoBMW
        let predictionPORSCHE = carInfoPorsche

        // show the prediction
        Common.printRed (sprintf "Single prediction:")
        Common.printRed (sprintf "  Predicted Price for %s-%s (%A) : %f" carInfoBMW.Make carInfoBMW.Model carInfoBMW.Year predictionBMW.Price)
        Common.printRed (sprintf "Single prediction:")
        Common.printRed (sprintf "  Predicted Price for %s-%s (%A) : %f" carInfoPorsche.Make carInfoPorsche.Model carInfoPorsche.Year predictionPORSCHE.Price)

