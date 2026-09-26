(* [Arange]'s exact int64 form (plan S6/T6.6/W): the same zero-input factory
   shape as [Arange_nwalk], but with [fmt = I64] and [exact = Some], drawing
   from a fixed base PAST 2^53 (matching
   [eval_symbolic_i64_arange_test.ml]'s own threshold) offset by [count] --
   every step stays exact-representable as int64 while still corrupting a
   float round trip, so the walk keeps proving exactness matters, not just
   Direct/Symbolic agreement on values small enough for either carrier. *)

module M = struct
  module W = Factory.Arange.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "arange_i64"
  let base = 9_007_199_254_740_993L

  let build pcg c =
    let count = W.count c in
    let stop = Int64.add base (Int64.of_int count) in
    let exact = { Factory.Arange.Exact.start = base; stop; step = 1L } in
    let params =
      {
        Factory.Arange.start = Int64.to_float base;
        stop = Int64.to_float stop;
        step = 1.;
        fmt = Payload.(Fmt I64);
        exact = Some exact;
      }
    in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"arange_i64" ~outputs:(fun r -> [ r ]) @@ arange params)
    in
    ({ Native_subject.target; graph = g; inputs = [] }, pcg)
end
