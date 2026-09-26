(* [Arange]'s default float form (plan S6/T6.6/W; design §2's own unwalked-op
   list): a zero-input factory like [Zeros]/[Eye] -- [Native_subject.t]
   itself needs no [~inputs] list entries. [0, count)] with a unit step,
   [fmt = F32], [exact = None] -- the plain ATen `arange(end)` shape, no
   int64 carrier involved (see [Arange_i64_nwalk] for that arm). *)

module M = struct
  module W = Factory.Arange.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "arange"

  let build pcg c =
    let count = W.count c in
    let params =
      {
        Factory.Arange.start = 0.;
        stop = float_of_int count;
        step = 1.;
        fmt = Payload.(Fmt F32);
        exact = None;
      }
    in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"arange" ~outputs:(fun r -> [ r ]) @@ arange params)
    in
    ({ Native_subject.target; graph = g; inputs = [] }, pcg)
end
