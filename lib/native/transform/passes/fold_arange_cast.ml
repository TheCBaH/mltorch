(* An integer [Arange] immediately cast to float becomes one [Arange(fmt=F32)]
   node. Value claim is [Identical], not a judgement call, ONLY when every
   generated value [start + i*step] is already integer-valued: [To_copy]'s own
   [Float] pixel is [S.load x out] unchanged (`pointwise_unary.ml`'s
   [Compute.pixel]), and [Arange]'s own pixel never reads [fmt] at all
   (`factory.ml`'s `S.add (S.const p.start) (S.mul (S.const p.step) ...)`), so
   the fused node's SYMBOLIC formula is textually the two-node chain's. But
   the two-node chain's actual materialized value is NOT that formula alone:
   `eval_direct.ml`'s [Arange] arm branches on the original I64 [fmt] to build
   an [int64] array via [Int64.of_float] (truncating toward zero), and the
   downstream [To_copy Float] then reads that truncated int64 back out as
   float (`Payload.get_float`'s [I64] case, [Int64.to_float]) -- a real
   float->int64->float round trip the fused F32 [Arange] never performs. For
   an integer [start]/[step] this round trip is a no-op (every generated value
   already has an exact integer float representation, well within the extent
   bound's headroom below 2^53), so the two computations agree -- but for a
   fractional [step] the truncation is a REAL transformation the fused node
   would silently skip. [start]/[step] are plain [float] fields with no
   type-level integer constraint (Native's Arange params are float regardless
   of the requested output [fmt] -- see the evaluator dtype design's "exact
   parameter representation" concern), so the pattern below guards this
   explicitly rather than assuming every I64-formatted Arange happens to have
   integer parameters.

   Unblocks EdgeNeXt's Fourier positional encoding, the corpus's only
   int-formatted factory whose sole consumer is a float cast: Kernel's
   [materializable] check requires every stage's own format to be f32 (the
   pixel evaluator only ever produces f32), so the raw i64 [Arange] stage was
   rejected before the cast that would have fixed it ever ran. See
   .ai/pt2_model_support.md. *)

open Graph_ir

let as_to_copy = function
  | To_copy (t : Pointwise.To_copy.t) -> Some t
  | _ -> None

(* [x]'s own value must be an exact integer, i.e. [Float.round x = x] --
   [Float.is_finite] is assumed already true here ([Arange.length] rejects a
   non-finite [start]/[step] before an [Arange] node can even exist). *)
let is_integer_valued x = Float.equal (Float.round x) x

let as_int_arange = function
  | Arange
      ({
         Factory.Arange.params =
           { fmt = Payload.Fmt Payload.I64; start; step; _ };
       } as a)
    when is_integer_valued start && is_integer_valued step ->
      Some a
  | _ -> None

(* [arange_id]/[to_copy_id] are removed; [anchor] (the to_copy's own output
   edge) is what the fused node's output preserves, so downstream consumers
   need no retargeting. *)
let pattern anchor =
  let open Pattern in
  let* (t : Pointwise.To_copy.t), to_copy_node = def anchor as_to_copy in
  let* () = guard (t.Pointwise.To_copy.target = Pointwise.To_copy.Float) in
  let* () = interior t.Pointwise.To_copy.x in
  let* (a : Factory.Arange.t), arange_node =
    def t.Pointwise.To_copy.x as_int_arange
  in
  return
    (anchor, arange_node.Node.id, to_copy_node.Node.id, a.Factory.Arange.params)

let build (anchor, arange_id, to_copy_id, (params : Factory.Arange.params))
    _region =
  let open Recipe in
  let* anchor = existing anchor in
  let f32_params =
    { params with Factory.Arange.fmt = Payload.Fmt Payload.F32 }
  in
  replace ~remove:[ arange_id; to_copy_id ]
    ~insert:
      [
        {
          op = Arange { Factory.Arange.params = f32_params };
          outputs = [ Preserved anchor ];
          from = [ arange_id; to_copy_id ];
        };
      ]
    ~claims:[ (anchor, Preserved anchor, Correspondence.Identical) ]
    ()

let pass =
  Pass.of_pattern ~name:"fold_arange_cast" ~pattern ~build:{ Pass.build }
