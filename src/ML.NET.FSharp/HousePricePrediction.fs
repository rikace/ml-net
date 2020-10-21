namespace ML.NET.FSharp

module HousePricePrediction =

    open System
    open Microsoft.ML
    open Microsoft.ML.Data
    open FSharp.Plotly

    /// The HouseBlockData class holds one single housing block data record.
    [<CLIMutable>]
    type HouseBlockData = {
        [<LoadColumn(0)>] Longitude : float32
        [<LoadColumn(1)>] Latitude : float32
        [<LoadColumn(2)>] HousingMedianAge : float32
        [<LoadColumn(3)>] TotalRooms : float32
        [<LoadColumn(4)>] TotalBedrooms : float32
        [<LoadColumn(5)>] Population : float32
        [<LoadColumn(6)>] Households : float32
        [<LoadColumn(7)>] MedianIncome : float32
        [<LoadColumn(8)>] MedianHouseValue : float32
    }

    /// The ToMedianHouseValue class is used in a column data conversion.
    [<CLIMutable>]
    type ToMedianHouseValue = {
        mutable NormalizedMedianHouseValue : float32
    }

    /// The ToRoomsPerPerson class is used in a column data conversion.
    [<CLIMutable>]
    type ToRoomsPerPerson = {
        mutable RoomsPerPerson : float32
    }

    /// The ToLocation class is used in a column data conversion.
    [<CLIMutable>]
    type FromLocation = {
        EncodedLongitude : float32[]
        EncodedLatitude : float32[]
    }

    /// The ToLocation class is used in a column data conversion.
    [<CLIMutable>]
    type ToLocation = {
        mutable Location : float32[]
    }

    /// file paths to data files (assumes os = windows!)
    let dataPath = Common.setPath "housing.csv"


    // create the machine learning context
    let context = MLContext(seed = Nullable 1)

    // load the dataset
    Common.printCyan "Load the data..."
    let data = context.Data.LoadFromTextFile<HouseBlockData>(dataPath, hasHeader = true, separatorChar = ',')

    // keep only records with a median house value < 500,000
    Common.printCyan "Filter data..."
    let dataFiltered = context.Data.FilterRowsByColumn(data, "MedianHouseValue", upperBound = 499999.0)

    // get an array of housing data
    let houses = context.Data.CreateEnumerable<HouseBlockData>(dataFiltered, reuseRowObject = false)

    // plot median house value by median income
    Common.printCyan "Plot median house value by median income"
    Chart.Point(houses |> Seq.map(fun h -> (h.MedianIncome, h.MedianHouseValue)))
        |> Chart.withX_AxisStyle "Median income"
        |> Chart.withY_AxisStyle "Median house value"
        |> Chart.Show


    // build a data loading pipeline
    let pipeline =
        // build a training pipeline
        Common.printCyan "Create pipeline..."
        EstimatorChain()

            // step 1: divide the median house value by 1000
            .Append(
                context.Transforms.CustomMapping(
                    Action<HouseBlockData, ToMedianHouseValue>(fun input output -> output.NormalizedMedianHouseValue <- input.MedianHouseValue / 1000.0f),
                    "MedianHouseValue"))

    let runMedianHouseValue () =
        Common.printYellow "Run MedianHouseValue..."

        // get a 10-record preview of the transformed data
        Common.printCyan "Training the model..."
        let model = dataFiltered |> pipeline.Fit
        let preview = (dataFiltered |> model.Transform).Preview(maxRows = 10)

        // show the preview
        Common.printCyan "Show the preview"
        preview.ColumnView |> Seq.iter(fun c ->
            printf "%-30s|" c.Column.Name
            preview.RowView |> Seq.iter(fun r -> printf "%10O|" r.Values.[c.Column.Index].Value)
            printfn "")

        Common.printCyan "Plot median house value by longitude"
        Chart.Point(houses |> Seq.map(fun h -> (h.Longitude, h.MedianHouseValue)))
            |> Chart.withX_AxisStyle "Longitude"
            |> Chart.withY_AxisStyle "Median house value"
            |> Chart.Show

    let runLocation() =
        Common.printYellow "Run Location..."
        // step 2: bin the longitude
        let locationPipeline =
            Common.printCyan "Create location-pipeline..."
            pipeline
                .Append(context.Transforms.NormalizeBinning("BinnedLongitude", "Longitude", maximumBinCount = 10))

                // step 3: bin the latitude
                .Append(context.Transforms.NormalizeBinning("BinnedLatitude", "Latitude", maximumBinCount = 10))

                // step 4: one-hot encode the longitude
                .Append(context.Transforms.Categorical.OneHotEncoding("EncodedLongitude", "BinnedLongitude"))

                // step 5: one-hot encode the latitude
                .Append(context.Transforms.Categorical.OneHotEncoding("EncodedLatitude", "BinnedLatitude"))

                .Append(
                    context.Transforms.CustomMapping(
                        Action<FromLocation, ToLocation>(fun input output ->
                            output.Location <- [|   for x in input.EncodedLongitude do
                                                        for y in input.EncodedLatitude do
                                                            x * y |] ),
                        "Location"))

        // get a 10-record preview of the transformed data
        Common.printCyan "Training the model..."
        let model = dataFiltered |> locationPipeline.Fit

        let preview = (dataFiltered |> model.Transform).Preview(maxRows = 10)

        Common.printCyan "Show the preview"
        preview.ColumnView |> Seq.iter(fun c ->
            printf "%-30s|" c.Column.Name
            preview.RowView |> Seq.iter(fun r -> printf "%10O|" r.Values.[c.Column.Index].Value)
            printfn "")

        Common.printCyan "Show the dense vector"
        preview.RowView |> Seq.iter(fun r ->
            let vector = r.Values.[r.Values.Length-1].Value :?> VBuffer<float32>
            vector.DenseValues() |> Seq.iter(fun v -> printf "%i" (int v))
            printfn "")

