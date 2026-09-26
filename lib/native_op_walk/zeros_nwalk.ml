(* [Zeros]'s first walk (plan S6/T6.6/W; design §2's own unwalked-op list):
   a zero-input factory -- [Native_subject.t] itself needs no [~inputs]
   list entries. F32 fixed by this walk file, not drawn (a factory has no
   operand to declare a format on). *)

module M = struct
  module W = Factory.Zeros.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "zeros"

  let build pcg c =
    let shape = W.shape c in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"zeros" ~outputs:(fun r -> [ r ])
          @@ zeros { Factory.Zeros.shape; fmt = Payload.(Fmt F32) })
    in
    ({ Native_subject.target; graph = g; inputs = [] }, pcg)
end
