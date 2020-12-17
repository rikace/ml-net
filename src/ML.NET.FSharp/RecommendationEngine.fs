namespace ML.NET.FSharp

module RecommendationEngine =

    open System
    open Microsoft.ML
    open Microsoft.ML.Data
    open Microsoft.ML.Recommender
    open Microsoft.ML.Trainers

    // Represents a product from the data-set
    [<CLIMutable>]
    type Product = {
        [<LoadColumn(0)>] ProductID : float32
        [<LoadColumn(1)>] [<ColumnName("Label")>] CombinedProductID : float32
    }

    [<CLIMutable>]
    type ProductPrediction = {
        [<ColumnName("Score")>] Score : float32
    }

    let dataPath = Common.setPath "Amazon0302.txt"

    // Step 1 - Create MLContext
    let context = MLContext(seed = Nullable 1)

    // Step 2 - Load the data
    Common.printCyan "Load the data..."

    let data = context.Data.LoadFromTextFile<Product>(dataPath, hasHeader = true, separatorChar = '\t')

    // Step 3 - Split the Data
    let dataPartitions = context.Data.TrainTestSplit(data, testFraction = 0.2)
    let trainSet = dataPartitions.TrainSet
    let testSet = dataPartitions.TestSet

    // Step 4 - Convert the Data /  Create Pipeline (define workflow)
    // prepare matrix factorization options
    let options = MatrixFactorizationTrainer.Options(   MatrixColumnIndexColumnName = "ProductIDEncoded",
                                                        MatrixRowIndexColumnName = "CombinedProductIDEncoded",
                                                        LabelColumnName = "Label",
                                                        LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass,
                                                        Alpha = 0.01,
                                                        Lambda = 0.025 )
    let pipeline =
        Common.printCyan "Create pipeline..."
        EstimatorChain()
         // map ProductID and CombinedProductID to keys
         .Append(context.Transforms.Conversion.MapValueToKey(inputColumnName = "ProductID", outputColumnName = "ProductIDEncoded"))
         .Append(context.Transforms.Conversion.MapValueToKey(inputColumnName = "Label", outputColumnName = "CombinedProductIDEncoded"))
         // find recommendations using matrix factorization
         .Append(context.Recommendation().Trainers.MatrixFactorization(options))

    // Step 5 - Training the algorithm for the model
    Common.printCyan "Training the model..."

    let model = trainSet |> pipeline.Fit

    // Step 6 - Use the test data against the model
    let metrics = testSet |> model.Transform |> context.Regression.Evaluate

    // show the metrics
    Common.printGreen (sprintf "Model metrics:")
    Common.printGreen (sprintf "  RMSE:%f" metrics.RootMeanSquaredError)
    Common.printGreen (sprintf "  MSE: %f" metrics.MeanSquaredError)

    // Step 8 - Use the model
    Common.printCyan "Create Prediction Engine"

    let engine : PredictionEngine<Product, ProductPrediction> = context.Model.CreatePredictionEngine model

    let runPrediction () =
        let productInfo = {
            ProductID = 21.f
            CombinedProductID = 77.f
        }

        let prediction = productInfo |> engine.Predict

        // show the prediction
        Common.printRed (sprintf "Score for product %f combined with %f is %f" productInfo.ProductID productInfo.CombinedProductID prediction.Score)

    let runPredictionBestMatches productId topN =
        let bestRecommendedProducts productId topN =
            seq {
                for index = 1 to 262110 do
                    let product =  {
                        ProductID = float32 productId
                        CombinedProductID = float32 index
                    }
                    let prediction = engine.Predict product
                    {| Score = prediction.Score; ProductID = index |}
            }
            |> Seq.sortByDescending(fun p -> p.Score)
            |> Seq.take topN

        for (index, product) in bestRecommendedProducts productId topN |> Seq.indexed do
            Common.printRed (sprintf "%d) Best match for product: %d is product: %d\t with score: %f" index productId product.ProductID product.Score)
