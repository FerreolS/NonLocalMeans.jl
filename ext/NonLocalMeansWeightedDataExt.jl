module NonLocalMeansWeightedDataExt
import NonLocalMeans: nonlocalmeans, nonlocalmeans!
import WeightedData: WeightedArray, get_value, get_precision


function nonlocalmeans(value::WeightedArray; kwargs...)
    out, out_precision = nonlocalmeans(
        get_value(value), get_precision(value); kwargs..., store_precision = true,
    )
    return WeightedArray(out, out_precision)
end

function nonlocalmeans!(output::WeightedArray, value::WeightedArray; kwargs...)
    outvalue = get_value(output)
    outprec = get_precision(output)
    nonlocalmeans!(outvalue, get_value(value), get_precision(value), outprec; kwargs...)
    return WeightedArray(outvalue, outprec)
end
end
