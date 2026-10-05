module NonLocalMeansWeightedDataExt
import NonlocalMeans: nonlocalmeans
import WeightedData: WeightedArray, get_value, get_precision


function nonlocalmeans(value::WeightedArray; kwargs...)
    return nonlocalmeans(get_value(value), get_precision(value); kwargs...)
end

function nonlocalmeans!(output::WeightedArray, value::WeightedArray; kwargs...)
    outvalue = get_value(output)
    outprec = get_precision(output)
    nonlocalmeans!(outvalue, get_value(value), get_precision(value), outprec; kwargs...)
    return WeightedArray(outvalue, outprec)
end
end
