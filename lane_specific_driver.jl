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
    agent # RL Agent
end
InterDriver(a::LaneSpecificAccelLatLon, s1::Int, s2::Int, s3::Int, action::Symbol, model_type::Symbol, p::Float64) = InterDriver(a, s1, s2, s3, action, model_type, p, nothing)

mapping = Dict(:DECEL => 1, :ZERO => 2, :ACCEL => 3)
unmapping = Dict(1 => :DECEL, 2 => :ZERO, 3 => :ACCEL)

flatten_state(s1, s2, s3) = (s1-1) * 121 + (s2-1)*11 + s3-1
function unflatten_state(f)
    s1 = 1 + div(f, 121)
    f -= (s1-1)*121
    s2 = 1 + div(f, 11)
    f -= (s2-1)*11
    s3 = f+1
    return (s1, s2, s3)
end


# This is a roundabout driver model that behaves correctly.
function get_safe_action(model)
    # If distance to another car is less than 2 and we are NOT in the roundabout
    if model.s2 < 2 && (model.s1 < L || model.s1 > 2*L)
        a = :DECEL
    # if we are going too fast, don't accelerate
    elseif model.s3 >= 10
        a = :ZERO
    # speed up to desired speed
    else
        a = :ACCEL
    end
    return a
end

# This is a dangerous driver who sometimes does not yield and sometimes stops in the roundabout!
# I see a lot of these drivers at Stanford which is why I decided to model them
function get_unsafe_action(model)
    # rand() returns a number between 0 and 1
    # Should we yield when outside the roundabout??
    if rand() < model.p && model.s2 < 2 && (model.s1 == 1 || model.s1 == 3)
        a = :DECEL
    # Should we stop inside the roundabout??
    elseif rand() > model.p && model.s2 < 2
        a = :DECEL
    # stay at desired speed
    elseif model.s3 >= 10
        a = :ZERO
    # speed up if too slow
    else
        a = :ACCEL
    end
    return a
end

function get_rl_action(model)
	s = flatten_state(model.s1, model.s2, model.s3)
	println("$(model.s1), $(model.s2), $(model.s3), $s")
	vals = model.agent.R[s,:] .+ model.agent.γ*model.agent.T[s,:]⋅model.agent.U
	# pick a random action to explore
	a = shuffle(findall(vals .== maximum(vals)))[1][2] # the last index [2] converts from CartesianIndex to linear in this special case
	# record progress
	model.agent.a_record[model.agent.ptr] = a
	model.agent.s_record[model.agent.ptr] = s
	model.agent.ptr += 1
    return unmapping[a]
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
	function is_on_collision_course(v1::Entity, v2::Entity)
		# idea: if velocity vectors intersect at a "close by" positive point we are going to collide
		vel1 = velg(v1.state)
		vel2 = velg(v2.state)
		if vel1.x == vel2.x && vel1.y == vel2.y # they are the same vehicle
			return false
		end
		#println([vel1.x vel2.x; vel1.y vel2.y] )
		#TODO fix
		δt = 0.1
		dx = (posg(v2.state)[1:2] .- posg(v1.state)[1:2])./δt
		cross_pt = [vel1.x -vel2.x-dx[1]; vel1.y -vel2.y-dx[2]] \ zeros(2)
		return all(cross_pt .> 0.0) && norm(cross_pt, 2) < 5.0
		#scene[egoid].state.posF.roadind.tag == scene[i].state.posF.roadind.tag
	end

    model.s1 = div(s, L) + 1 # each lane is approximately 3L long so this divides into 1, 2, 3
    states = map( i -> posg(scene[i].state)[1:2] , filter( (i) -> is_on_collision_course(scene[egoid], scene[i]), 1:length(scene)) )
    if length(states) == 0
        model.s2 = 10
    else
        model.s2 = max(10, Integer(trunc(0.2*minimum(map( (s) -> norm(x .- s, 2), states) ))))
    end
    v = velf(scene[egoid].state)
    model.s3 = max(10, Integer(trunc(norm([v.t; v.s], 2))))
    
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