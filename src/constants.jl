using FastAI, Plots, Flux, StaticArrays, LearnBase, Mongoc, CSV, DataFrames, Distributions
using Mongoc: BSONObjectId, BSON

# const REFERENCE_RANGE = TODO

const GOLD = BSONObjectId("5f5a83183c9d9fd8800ce8a3")
const VACUUM = BSONObjectId("5f5a831c3c9d9fd8800ce92c")
const BATCH_SIZE = 1_024

const NUM_WAVELENS = 150
const NUM_SIMULATORS = 1

# const MIN_CLIP = 1e-19
const MIN_CLIP = 1e-14

const MAX_SCATTER_CUTOFF = 1e-2
const MIN_SCATTER_CUTOFF = 1e-14

const MAX_ABSORPTION_CUTOFF = 1e-11
const BULK = DataFrame(CSV.File("data/gold_bulk_emissivity.csv"))
