using AutomotiveSimulator
using Random
using LinearAlgebra

# from https://sisl.github.io/AutomotiveSimulator.jl/dev/tutorials/intersection/
# We will use lateral and longitudinal acceleration to control a car in the intersection. The first step is to define a corresponding action type that will contain the acceleration inputs.

struct LaneSpecificAccelLatLon
    a_lat::Float64
    a_lon::Float64
end

# Next, add a method to the propagate function to update the state using our new action type.

function AutomotiveSimulator.propagate(veh::Entity, action::LaneSpecificAccelLatLon, roadway::Roadway, Δt::Float64)
    lane_tag_orig = veh.state.posF.roadind.tag
    state = propagate(veh, LatLonAccel(action.a_lat, action.a_lon), roadway, Δt)
    roadproj = proj(state.posG, roadway[lane_tag_orig], roadway, move_along_curves=false)
    retval = VehicleState(Frenet(roadproj, roadway), roadway, state.v)
    return retval
end

# We define a driver model, which can be seen as a distribution over actions. # TODO Here we will define the simplest model, which is to repeat the same action.
mutable struct InterDriver <: DriverModel{LaneSpecificAccelLatLon}
    a::LaneSpecificAccelLatLon
    s1::Int # state 1 (distance traveled (in buckets))
    s2::Int # state 2 (distance to other vehicle (in buckets))
    s3::Int # velocity (in buckets)
    action::Symbol # :DECEL, :ZERO or :ACCEL
    model_type::Symbol # :RL, :SAFE = fixed safe policy, :UNSAFE = unsafe policy
    p::Float64 # Threshold used for unsafe policy to decide whether to be dangerous. Setting this high produces safer behavior
end

# This is a roundabout driver model that behaves correctly.
function get_safe_action(model)
    # If distance to another car is less than 2 and we are NOT in the roundabout
    if model.s2 < 2 && (model.s1 < L || model.s1 > 2*L)
        return :DECEL
    # if we are going too fast, don't accelerate
    elseif model.s3 >= 10
        return :ZERO
    # speed up to desired speed
    else
        return :ACCEL
    end
end

# This is a dangerous roundabout driver who sometimes does not yield and sometimes stops in the intersection!
# I see a lot of these drivers at Stanford which is why I decided to model them
function get_unsafe_action(model)
    # rand() returns a number between 0 and 1
    # Should we yield when outside the roundabout??
    if rand() < model.p && model.s2 < 2 && (model.s1 == 1 || model.s1 == 3)
        return :DECEL
    # Should we stop inside the roundabout??
    elseif rand() < model.p && model.s2 < 2
        return :DECEL
    # stay at desired speed
    elseif model.s3 >= 10
        return :ZERO
    # speed up if too slow
    else
        return :ACCEL
    end
end

function get_action(model::InterDriver)
    if model.model_type == :RL
        return get_rl_action(model)
    elseif model.model_type == :SAFE
        return get_safe_action(model)
    elseif model.model_type == :UNSAFE
        return get_unsafe_action(model)
    end
end


# update the state with observations here
function AutomotiveSimulator.observe!(model::InterDriver, scene::Scene, roadway::Roadway, egoid::Int64)
    
    # update state
    x = posg(scene[egoid].state)[1:2]
    v = velg(scene[egoid].state)
    s = posf(scene[egoid]).s
    model.s1 = div(s, L) # each lane is approximately 3L long so this divides into 1, 2, 3
    states = map( i -> posg(scene[i].state)[1:2] , filter( (i) -> i != egoid && (posg(scene[i].state)[1:2]-x)⋅v > 0.0, 1:length(scene)) )
    if length(states) == 0
        model.s2 = 10
    else
        model.s2 = Integer(trunc(0.2*minimum(map( (s) -> norm(x .- s, 2), states) )))
    end
    v = velf(scene[egoid].state)
    model.s3 = Integer(trunc(norm([v.t; v.s], 2)))
    
    # get action
    model.action = get_action(model)
    # update based on action
    th = posg(scene[egoid].state)[3]
    # translate high level MDP action to accel signal
    if model.action == :DECEL
        model.a = LaneSpecificAccelLatLon(-v.t, -v.s)
    elseif model.action == :ZERO
        model.a = LaneSpecificAccelLatLon(0.0, 0.0)
    elseif model.action == :ACCEL
        model.a = model.a = LaneSpecificAccelLatLon(0.1*v.t, 0.5*v.s)
    end
    return model
end

# Samples an action from the model
function Base.rand(::AbstractRNG, model::InterDriver)
    return model.a
end