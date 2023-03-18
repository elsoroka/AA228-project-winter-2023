# Construct the roundabout

using AutomotiveSimulator

w = DEFAULT_LANE_WIDTH
L=30
r = 8.0
function append_to_curve!(target::Curve, newstuff::Curve)
    s_end = target[end].s
    for c in newstuff
        push!(target, CurvePt(c.pos, c.s+s_end, c.k, c.kd))
    end
    return target
end

function make_roundabout()
	roadway = Roadway()
	# make a loop

	curve = gen_bezier_curve(VecSE2(0,0.0),VecSE2(r,r,0.5*π), 0.6r, 0.6r, 51)[2:end]
	append_to_curve!(curve, gen_bezier_curve(VecSE2(r,r,0.5*π), VecSE2(0,2*r, π),  0.6r, 0.6r, 51)[2:end])
	append_to_curve!(curve, gen_bezier_curve(VecSE2(0,2*r,π), VecSE2(-r,r, 1.5*π),  0.6r, 0.6r, 51)[2:end])
	append_to_curve!(curve, gen_bezier_curve(VecSE2(-r,r, 1.5*π), VecSE2(0,0.0),  0.6r, 0.6r, 51)[2:end])

	lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
	push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

	# Add one in-and-out to the roundabout
	curve = gen_straight_curve(convert(VecE2, VecSE2(0-w/2,3r+L,0)), convert(VecE2, VecSE2(0-w/2,3r,π)), 2)
	curve = append_to_curve!(curve, gen_bezier_curve(VecSE2(0-w/2,3r,-0.5*π), VecSE2(-r,r,-0.5π), 1.1r, 0.9r, 51)[2:end])
	curve = append_to_curve!(curve, gen_bezier_curve(VecSE2(-r,r,-0.5π), VecSE2(0-w/2,-r,-0.5*π), 0.9r, 1.1r, 51))
	curve = append_to_curve!(curve, gen_straight_curve(convert(VecE2, VecSE2(0-w/2,-r,0)), convert(VecE2, VecSE2(0-w/2,-r-L,0)), 2))


	lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
	push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

	# Add second in-and-out to the roundabout

	curve = gen_straight_curve(convert(VecE2, VecSE2(w/2,3r+L,0)), convert(VecE2, VecSE2(w/2,3r,π)), 2)
	curve = append_to_curve!(curve, gen_bezier_curve(VecSE2(w/2,3r,-0.5*π), VecSE2(r,r,-0.5π), 1.1r, 0.9r, 51)[2:end])
	curve = append_to_curve!(curve, gen_bezier_curve(VecSE2(r,r,-0.5π), VecSE2(w/2,-r,-0.5*π), 0.9r, 1.1r, 51))
	curve = append_to_curve!(curve, gen_straight_curve(convert(VecE2, VecSE2(w/2,-r,0)), convert(VecE2, VecSE2(w/2,-r-L,0)), 2))


	lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
	push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

	# Add third (horizontal) in-and-out to the roundabout
	curve = gen_straight_curve(convert(VecE2, VecSE2(-2r-L,r-w/2,0)), convert(VecE2, VecSE2(-2r,r-w/2, 0)), 2)
	curve = append_to_curve!(curve, gen_bezier_curve(VecSE2(-2r,r-w/2,0), VecSE2(0,0.,0), 1.1r, 0.9r, 51)[2:end])
	curve = append_to_curve!(curve, gen_bezier_curve(VecSE2(0,0.,0), VecSE2(2r,r-w/2,0), 0.9r, 1.1r, 51))
	curve = append_to_curve!(curve, gen_straight_curve(convert(VecE2, VecSE2(2r,r-w/2,0)), convert(VecE2, VecSE2(2r+L,r-w/2,0)), 2))

	lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
	push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

	# Add fourth (horizontal) in-and-out to the roundabout

	curve = gen_straight_curve(convert(VecE2, VecSE2(-2r-L,r+w/2,0)), convert(VecE2, VecSE2(-2r,r+w/2, 0)), 2)
	curve = append_to_curve!(curve, gen_bezier_curve(VecSE2(-2r,r+w/2,0), VecSE2(0,2r,0), 1.1r, 0.9r, 51)[2:end])
	curve = append_to_curve!(curve, gen_bezier_curve(VecSE2(0,2r,0), VecSE2(2r,r+w/2,0), 0.9r, 1.1r, 51))
	curve = append_to_curve!(curve, gen_straight_curve(convert(VecE2, VecSE2(2r,r+w/2,0)), convert(VecE2, VecSE2(2r+L,r+w/2,0)), 2))

	lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
	push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))


	return roadway
end