(* [Eye]'s first walk (plan S6/T6.6/W); see [Zeros_nwalk]'s own comment for
   the rationale, identical here. *)

module M = struct
  module W = Factory.Eye.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "eye"

  let build pcg c =
    let shape = W.shape c in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"eye" ~outputs:(fun r -> [ r ])
          @@ eye { Factory.Eye.shape; fmt = Payload.(Fmt F32) })
    in
    ({ Native_subject.target; graph = g; inputs = [] }, pcg)
end
