########################################
##    Iterator version of cumsum()

"Implements 1D `cumsum()` as an iterable"
struct iter_cumsum{TElements}
    elements::TElements
end

Base.IteratorEltype(::Type{<:iter_cumsum{TElements}}) where {TElements} = Base.IteratorEltype(TElements)
Base.eltype(::Type{<:iter_cumsum{TElements}}) where {TElements} = let et = eltype(TElements)
    if et == Union{}
        Union{}
    else
        Union{Base.return_types(+, (et, et))...}
    end
end

Base.IteratorSize(::Type{<:iter_cumsum{TElements}}) where {TElements} = let i = Base.IteratorSize(TElements)
    if i isa Union{Base.SizeUnknown, Base.IsInfinite}
        i
    elseif i isa Union{Base.HasLength, Base.HasShape}
        Base.HasLength()
    else
        error("Unhandled: ", i)
    end
end
Base.length(ic::iter_cumsum) = length(ic.elements)

function Base.iterate(ic::iter_cumsum)
    first_iter = iterate(ic.elements)
    if isnothing(first_iter)
        return nothing
    end

    (first_el, first_state) = first_iter
    return (first_el, (first_el, first_state))
end
function Base.iterate(ic::iter_cumsum, (prev_el, prev_state))
    next_iter = iterate(ic.elements, prev_state)
    if isnothing(next_iter)
        return nothing
    end

    (next_el, next_state) = next_iter
    next_val = next_el + prev_el
    return (next_val, (next_val, next_state))
end

export iter_cumsum


########################################
##    Iterator version of diff()

"Implements 1D `diff()` as an iterable"
struct iter_diff{TElements}
    elements::TElements
end
iter_diff(elements) = iter_diff{typeof(elements)}(elements)

Base.IteratorEltype(::Type{<:iter_diff{TElements}}) where {TElements} = Base.IteratorEltype(TElements)
Base.eltype(::Type{<:iter_diff{TElements}}) where {TElements} = let et = eltype(TElements)
    if et == Union{}
        Union{}
    else
        Union{Base.return_types(-, (et, et))...}
    end
end

Base.IteratorSize(::Type{<:iter_diff{TElements}}) where {TElements} = let i = Base.IteratorSize(TElements)
    if i isa Union{Base.SizeUnknown, Base.IsInfinite}
        i
    elseif i isa Union{Base.HasLength, Base.HasShape}
        Base.HasLength()
    else
        error("Unhandled: ", i)
    end
end
Base.length(id::iter_diff) = max(0, length(id.elements) - 1)

function Base.iterate(id::iter_diff{TElements}) where {TElements}
    inner_first_iter = iterate(id.elements)
    if isnothing(inner_first_iter)
        return nothing
    end
    (inner_first_el, inner_first_state) = inner_first_iter

    inner_second_iter = iterate(id.elements, inner_first_state)
    if isnothing(inner_second_iter)
        return nothing
    end
    (inner_second_el, inner_second_state) = inner_second_iter

    return (inner_second_el - inner_first_el, (inner_second_el, inner_second_state))
end
function Base.iterate(id::iter_diff{TElements}, (inner_el, inner_state)) where {TElements}
    inner_next_iter = iterate(id.elements, inner_state)
    if isnothing(inner_next_iter)
        return nothing
    end
    (inner_next_el, inner_next_state) = inner_next_iter

    return (inner_next_el - inner_el, (inner_next_el, inner_next_state))
end

export iter_diff