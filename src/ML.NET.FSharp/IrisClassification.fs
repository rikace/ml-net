namespace ML.NET.FSharp


module IrisClassification =
    open System
    open Microsoft.ML
    open Microsoft.ML.Data

    /// A type that holds a single iris flower.
    [<CLIMutable>]
    type IrisData = {
        [<LoadColumn(0)>] SepalLength : float32
        [<LoadColumn(1)>] SepalWidth : float32
        [<LoadColumn(2)>] PetalLength : float32
        [<LoadColumn(3)>] PetalWidth : float32
        [<LoadColumn(4)>] Label : string
    }

    /// A type that holds a single model prediction.
    [<CLIMutable>]
    type IrisPrediction = {
        PredictedLabel : uint32
        Score : float32[]
    }

    let dataPath = Common.setPath "iris-data.csv"

    // get the machine learning context
    let context = new MLContext(seed = Nullable 1)

    // read the iris flower data from a text file
    Common.printCyan "Load the data..."
    let data = context.Data.LoadFromTextFile<IrisData>(dataPath, hasHeader = false, separatorChar = ',')

    // split the data into a training and testing partition
    let partitions = context.Data.TrainTestSplit(data, testFraction = 0.2)

    // set up a learning pipeline
    let pipeline =
        // build a training pipeline
        Common.printCyan "Create pipeline..."
        EstimatorChain()
            // step 1: concatenate features into a single column
            .Append(context.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))

            // step 2: use k-means clustering to find the iris types
            .Append(context.Clustering.Trainers.KMeans(numberOfClusters = 3))

    // train the model on the training data
    Common.printCyan "Training the model..."
    let model = partitions.TrainSet |> pipeline.Fit

    // get predictions and compare to ground truth
    Common.printCyan "Training the model..."
    let metrics = partitions.TestSet |> model.Transform |> context.Clustering.Evaluate

    // show results
    Common.printGreen (sprintf "Nodel results")
    Common.printGreen (sprintf "   Average distance:     %f" metrics.AverageDistance)
    Common.printGreen (sprintf "   Davies Bouldin index: %f" metrics.DaviesBouldinIndex)

    // set up a prediction engine
    Common.printCyan "Create Prediction Engine"
    let engine = context.Model.CreatePredictionEngine model

    // grab 3 flowers from the dataset
    Common.printYellow "Model predictions:"
    let flowers = context.Data.CreateEnumerable<IrisData>(partitions.TestSet, reuseRowObject = false) |> Array.ofSeq

    let runPrediction () =
        // Test data for prediction
        let testInstance : IrisData = {
            SepalLength = 3.3f
            SepalWidth = 1.6f
            PetalLength = 0.2f
            PetalWidth = 5.1f
            Label = "Iris-setosa"
        }

        let testFlowers = [ flowers.[0]; flowers.[10]; flowers.[20]; testInstance ]

        // show predictions for the three flowers
        Common.printGreen "Predictions for the 4 test flowers:"
        Common.printGreen "  Label\t\t\tPredicted\tScores"
        testFlowers |> Seq.iter(fun f ->
                let p = engine.Predict f
                printf "  %-15s\t%i\t\t" f.Label p.PredictedLabel
                p.Score |> Seq.iter(fun s -> printf "%f\t" s)
                printfn "")
