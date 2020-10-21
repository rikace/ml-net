namespace ML.NET.FSharp

module TitanicPrediction =

    open System
    open System.IO
    open Microsoft.ML
    open Microsoft.ML.Data
    open Microsoft.ML.Transforms

    /// The Passenger class represents one passenger on the Titanic.
    [<CLIMutable>]
    type Passenger = {
        [<LoadColumn(1)>] Label : bool
        [<LoadColumn(2)>] Pclass : float32
        [<LoadColumn(4)>] Sex : string
        [<LoadColumn(5)>] RawAge : string // not a float!
        [<LoadColumn(6)>] SibSp : float32
        [<LoadColumn(7)>] Parch : float32
        [<LoadColumn(8)>] Ticket : string
        [<LoadColumn(9)>] Fare : float32
        [<LoadColumn(10)>] Cabin : string
        [<LoadColumn(11)>] Embarked : string
    }

    /// The PassengerPrediction class represents one model prediction.
    [<CLIMutable>]
    type PassengerPrediction = {
        // when we do supervise learning to predict a value, we deal with label data.
        // this vale "Prediction" is the outcome, is the label, the answer to our model
        // By attributing this field as Label we notify ML.NET that this is the column containing
        // the values that we are going to try to predict and that these values are correlated to the input values
        [<ColumnName("PredictedLabel")>] Prediction : bool
        Probability : float32
        Score : float32
    }

    /// The ToAge class is a helper class for a column transformation.
    [<CLIMutable>]
    type Age = {
        mutable Age : string
    }

    let trainDataPath = Common.setPath "titanic/train_data.csv"

    let testDataPath = Common.setPath "titanic/test_data.csv"


    // set up a machine learning context
    let context = new MLContext(seed = Nullable 1)

    // load the training and testing data in memory
    Common.printCyan "Loading the data..."
    let trainData : IDataView = context.Data.LoadFromTextFile<Passenger>(trainDataPath, hasHeader = true, separatorChar = ',', allowQuoting = true)
    let testData : IDataView = context.Data.LoadFromTextFile<Passenger>(testDataPath, hasHeader = true, separatorChar = ',', allowQuoting = true)


    // set up a training pipeline
    let pipeline =
        Common.printCyan  "Creating pipeline..."
        EstimatorChain()

            // step 1: replace missing ages with '?'
            .Append(
                context.Transforms.CustomMapping(
                    Action<Passenger, Age>(fun input output -> output.Age <- if String.IsNullOrEmpty(input.RawAge) then "?" else input.RawAge),
                    "AgeMapping"))

            // step 2: convert string ages to floats
            .Append(context.Transforms.Conversion.ConvertType("Age", outputKind = DataKind.Single))

            // step 3: replace missing age values with the mean age
            .Append(context.Transforms.ReplaceMissingValues("Age", replacementMode = MissingValueReplacingEstimator.ReplacementMode.Mean))

            // step 4: replace string columns with one-hot encoded vectors
            .Append(context.Transforms.Categorical.OneHotEncoding("Sex"))
            .Append(context.Transforms.Categorical.OneHotEncoding("Ticket"))
            .Append(context.Transforms.Categorical.OneHotEncoding("Cabin"))
            .Append(context.Transforms.Categorical.OneHotEncoding("Embarked"))

            // step 5: concatenate everything into a single feature column
            .Append(context.Transforms.Concatenate("Features", "Age", "Pclass", "SibSp", "Parch", "Sex", "Embarked"))


            // step 6: use a BinaryClassification (fast-tree) trainer
            .Append(context.BinaryClassification.Trainers.FastTree())
            //.Append(context.BinaryClassification.Trainers.FastTree(numberOfLeaves = 150, numberOfTrees = 150, minimumExampleCountPerLeaf = 1)) // leaves and tree
            //.Append(context.BinaryClassification.Trainers.LbfgsLogisticRegression())
            //.Append(context.BinaryClassification.Trainers.SdcaLogisticRegression())


    // train the model
    Common.printCyan "Training the data..."
    let model = trainData |> pipeline.Fit

    // make predictions and compare with ground truth
    Common.printCyan "Evaluate the model..."
    let metrics = testData |> model.Transform |> context.BinaryClassification.Evaluate

    // report the results
    Common.printYellow "Model metrics:"
    Common.printGreen (sprintf "  Accuracy:          %f" metrics.Accuracy)
    Common.printGreen (sprintf "  Auc:               %f" metrics.AreaUnderRocCurve)
    Common.printGreen (sprintf "  Auprc:             %f" metrics.AreaUnderPrecisionRecallCurve)
    Common.printGreen (sprintf "  F1Score:           %f" metrics.F1Score)
    Common.printGreen (sprintf "  LogLoss:           %f" metrics.LogLoss)
    Common.printGreen (sprintf "  LogLossReduction:  %f" metrics.LogLossReduction)
    Common.printGreen (sprintf "  PositivePrecision: %f" metrics.PositivePrecision)
    Common.printGreen (sprintf "  PositiveRecall:    %f" metrics.PositiveRecall)
    Common.printGreen (sprintf "  NegativePrecision: %f" metrics.NegativePrecision)
    Common.printGreen (sprintf "  NegativeRecall:    %f" metrics.NegativeRecall)

    // set up a prediction engine
    Common.printCyan "Create Prediction Engine"
    let engine = context.Model.CreatePredictionEngine model

    let runPrediction() =
        // create sample records
        let passenger1 = {
            Pclass = 1.0f
            Sex = "female"
            RawAge = "48"
            SibSp = 0.0f
            Parch = 0.0f
            Ticket = "B"
            Fare = 30.0f
            Cabin = "123"
            Embarked = "S"
            Label = false // unused!
        }
        let passenger2 = {
            Pclass = 2.0f
            Sex = "male"
            RawAge = "72"
            SibSp = 0.0f
            Parch = 0.0f
            Ticket = "B"
            Fare = 70.0f
            Cabin = "123"
            Embarked = "S"
            Label = false // unused!
        }
        let passenger3 = {
            Pclass = 3.0f
            Sex = "male"
            RawAge = "2"
            SibSp = 0.0f
            Parch = 0.0f
            Ticket = "B"
            Fare = 45.0f
            Cabin = "123"
            Embarked = "S"
            Label = false // unused!
        }
        let passenger4 = {
            Pclass = 2.0f
            Sex = "female"
            RawAge = "72"
            SibSp = 0.0f
            Parch = 0.0f
            Ticket = "B"
            Fare = 80.0f
            Cabin = "123"
            Embarked = "S"
            Label = false // unused!
        }



        // report the results
        Common.printCyan "Model prediction:"
        for passenger in [passenger1;passenger2;passenger3;passenger4] do
            // make the prediction
            let prediction = engine.Predict passenger
            let color = if prediction.Prediction then ConsoleColor.Green else ConsoleColor.Red
            Common.printColor color (sprintf "  Passenger Class %A - Gender %A - Age %A - Fare %A" passenger.Pclass passenger.Sex passenger.RawAge passenger.Fare)
            Common.printColor color (sprintf "  Prediction:  %s" (if prediction.Prediction then "survived" else "perished"))
            Common.printColor color (sprintf "  Probability: %f" prediction.Probability)

