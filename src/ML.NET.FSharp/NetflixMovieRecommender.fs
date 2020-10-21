namespace ML.NET.FSharp

module NetflixMovieRecommender =

    open System
    open Microsoft.ML
    open Microsoft.ML.Trainers
    open Microsoft.ML.Data

    /// The MovieRating class holds a single movie rating.
    [<CLIMutable>]
    type MovieRating = {
        [<LoadColumn(0)>] UserID : float32
        [<LoadColumn(1)>] MovieID : float32
        [<LoadColumn(2)>] Label : float32
    }

    /// The MovieRatingPrediction class holds a single movie prediction.
    [<CLIMutable>]
    type MovieRatingPrediction = {
        Label : float32
        Score : float32
    }

    /// The MovieTitle class holds a single movie title.
    [<CLIMutable>]
    type MovieTitle = {
        [<LoadColumn(0)>] MovieID : float32
        [<LoadColumn(1)>] Title : string
        [<LoadColumn(2)>] Genres: string
    }

    // file paths to data files (assumes os = windows!)
    let trainDataPath = Common.setPath "recommendation/recommendation-ratings-train.csv"
    let testDataPath = Common.setPath "recommendation/recommendation-ratings-test.csv"
    let titleDataPath = Common.setPath "recommendation/recommendation-movies.csv"


    // set up a new machine learning context
    let context = MLContext(seed = Nullable 1)

    // load training and test data
    Common.printCyan "Load the data..."
    let trainData = context.Data.LoadFromTextFile<MovieRating>(trainDataPath, hasHeader = true, separatorChar = ',')
    let testData = context.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader = true, separatorChar = ',')

    // prepare matrix factorization options
    let options =
        MatrixFactorizationTrainer.Options(
            MatrixColumnIndexColumnName = "UserIDEncoded",
            MatrixRowIndexColumnName = "MovieIDEncoded",
            LabelColumnName = "Label",
            NumberOfIterations = 20,
            ApproximationRank = 100)

    // set up a training pipeline
    let pipeline =
        // build a training pipeline
        Common.printCyan "Create pipeline..."
        EstimatorChain()


            // step 1: map userId and movieId to keys
            .Append(context.Transforms.Conversion.MapValueToKey("UserIDEncoded", "UserID"))
            .Append(context.Transforms.Conversion.MapValueToKey("MovieIDEncoded", "MovieID"))

            // step 2: find recommendations using matrix factorization
            .Append(context.Recommendation().Trainers.MatrixFactorization(options))

    // train the model
    Common.printCyan "Training the model..."
    let model = trainData |> pipeline.Fit

    // calculate predictions and compare them to the ground truth
    let metrics = testData |> model.Transform |> context.Recommendation().Evaluate

    Common.printGreen (sprintf "Evaluation metrics")
    // show model metrics
    Common.printCyan "Model metrics:"
    Common.printCyan (sprintf "  RMSE: %f" metrics.RootMeanSquaredError)
    Common.printCyan (sprintf "  MAE:  %f" metrics.MeanAbsoluteError)
    Common.printCyan (sprintf "  MSE:  %f" metrics.MeanSquaredError)

    // set up a prediction engine
    Common.printCyan "Create Prediction Engine"
    let engine = context.Model.CreatePredictionEngine model

    let runPrediction() =
        // load all movie titles
        let movieData = context.Data.LoadFromTextFile<MovieTitle>(titleDataPath, hasHeader = true, separatorChar = ',', allowQuoting = true)
        let movies = context.Data.CreateEnumerable(movieData, reuseRowObject = false)

        let marksMovies =
            movies |> Seq.map(fun m ->
                let p2 = engine.Predict { UserID = 712.0f; MovieID = m.MovieID; Label = 0.0f }
                (m.Title, p2.Score))
            |> Seq.sortByDescending snd

        // print the results
        Common.printRed "What are Ricky's top-3 movies?"
        marksMovies |> Seq.take(3) |> Seq.iter(fun t -> Common.printYellow (sprintf "  %f %s" (snd t) (fst t)))


        // 2571,"Matrix, The (1999)",Action|Sci-Fi|Thriller
        Common.printMagenta "Do I Like \"Matrix\" ?"
        let predict1 = engine.Predict { UserID = 712.0f; MovieID = 2571.0f; Label = 0.0f }
        Common.printGreen (sprintf "  Score: %f" predict1.Score)


        // 2982,", The (1990)",Horror|Thriller
        Common.printMagenta "Do I Like \"Guardian of the Galaxy\" ?"
        let predict2 = engine.Predict { UserID = 712.0f; MovieID = 2982.0f; Label = 0.0f }
        Common.printGreen (sprintf "  Score: %f" predict2.Score)



