(* See eval_direct_compute.mli. *)

open Graph_ir
module E = Eval_op.Make (Direct)

type error =
  [ `Arange_i64_overflow of Factory.Arange.Overflow.t
  | `Unsupported_to_copy_bool_source of Payload.packed_fmt
  | `Unsupported_to_copy_long_source of Payload.packed_fmt ]

let pp_error ppf : [< error ] -> unit = function
  | `Arange_i64_overflow { Factory.Arange.Overflow.start; step; i } ->
      Format.fprintf ppf
        "arange: exact int64 generation overflows at start=%Ld step=%Ld i=%d"
        start step i
  | `Unsupported_to_copy_bool_source (Payload.Fmt f) ->
      Format.fprintf ppf
        "to_copy: Bool target has no exact Bool output for a %s source"
        (Payload.fmt_name f)
  | `Unsupported_to_copy_long_source (Payload.Fmt f) ->
      Format.fprintf ppf
        "to_copy: Long target has no exact I64 output for a %s source"
        (Payload.fmt_name f)

(* An index-style output landed as exact int64 storage. *)
let index_i64 out_shape ~x_shape ~x pixel =
  Tensor.materialize_i64 out_shape (fun coord ->
      Int64.of_float (pixel ~x_shape ~x coord))

let compute (g : graph) (op : op) ~(output : Output_ordinal.t) ~out_shape
    ~operand_env ~shape_env ~fill : (Tensor.packed, [> error ]) Err.t =
  match op with
  | Unbind { Split.Unbind.params; x } ->
      Err.return
        (Tensor.unbind
           (Tensor_id.Map.find x operand_env)
           ~axis:params.axis ~output ~shape:out_shape)
  (* Same dtype-preserving bypass as [Unbind], and for the same reason:
     [offset] is the sum of every earlier piece's size, computed the same way
     [Eval_op]'s arm computes it for the generic path. *)
  | Split_with_sizes { Split.Split_with_sizes.params; x } ->
      let offset =
        Split.Split_with_sizes.offset_of ~output
          params.Split.Split_with_sizes.sizes
      in
      Err.return
        (Tensor.split_with_sizes
           (Tensor_id.Map.find x operand_env)
           ~axis:params.Split.Split_with_sizes.axis ~offset ~shape:out_shape)
  | Zeros { Factory.Zeros.params } ->
      Err.return (Tensor.materialize_fmt params.fmt out_shape (fun _ -> 0.))
  | Eye { Factory.Eye.params } ->
      Err.return
        (Tensor.materialize_fmt params.fmt out_shape (fun coord ->
             if Dim.to_int coord.Vec6.w = Dim.to_int coord.Vec6.c then 1.
             else 0.))
  | Arange { Factory.Arange.params } -> (
      match params.fmt with
      | Payload.Fmt Payload.I64 -> (
          match params.exact with
          (* Exact int64 arithmetic, no float round trip: fixes the
             truncation [Int64.of_float (Arange.value ...)] below performs
             whenever an ATen call actually supplied exact integer bounds. *)
          | Some e ->
              Err.Escape.with_escape (fun esc ->
                  Tensor.materialize_i64 out_shape (fun coord ->
                      Err.Escape.or_throw esc
                        (Factory.Arange.value_i64_exact e
                           (Dim.to_int coord.Vec6.c))))
          | None ->
              Err.return
                (Tensor.materialize_i64 out_shape (fun coord ->
                     Int64.of_float
                       (Factory.Arange.value params (Dim.to_int coord.Vec6.c))))
          )
      | _ ->
          Err.return
            (Tensor.materialize_fmt params.fmt out_shape (fun coord ->
                 Factory.Arange.value params (Dim.to_int coord.Vec6.c))))
  (* Dtype-preserving Reshape: the default arm below reaches
     [Reshape.Reshape.Compute(Direct).pixel], whose final [S.load] reads
     through [Payload.get_float] regardless of the source format -- exact
     for F32 but silently lossy above 2^53 for an I64 source. Branching on
     the OPERAND's declared signature format (not the runtime payload,
     though they agree by construction) routes an I64 reshape through
     [Compute_i64]/[Tensor.i64_load] instead, matching this node's Arange
     arm just above. Every other format keeps the existing float pixel
     path, unchanged. *)
  | Reshape { Reshape.Reshape.params; x } -> (
      let x_sig = Tensor_id.Map.find x g.Graph.tensors in
      match x_sig.Tensor_sig.fmt with
      | Payload.Fmt Payload.I64 ->
          let module C = Reshape.Reshape.Compute_i64 (Direct) (Direct) in
          let x_t = Tensor_id.Map.find x operand_env in
          let x_shape = Tensor_id.Map.find x shape_env in
          Err.return
            (Tensor.materialize_i64 out_shape (fun coord ->
                 C.pixel params ~x_shape ~x:x_t coord))
      | _ ->
          Err.return
            (Schedule.evaluate out_shape
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  (* Dtype-preserving Permute, the same shape as Reshape just above: the
     default arm's [Permute.Compute(S).pixel] reads through [S.load], exact
     for F32 but silently lossy above 2^53 for an I64 source. Branch on the
     OPERAND's declared format, matching Reshape/Arange's own precedent. *)
  | Permute { Permute.Permute.perm; x } -> (
      let x_sig = Tensor_id.Map.find x g.Graph.tensors in
      match x_sig.Tensor_sig.fmt with
      | Payload.Fmt Payload.I64 ->
          let module C = Permute.Permute.Compute_i64 (Direct) (Direct) in
          let x_t = Tensor_id.Map.find x operand_env in
          Err.return
            (Tensor.materialize_i64 out_shape (fun coord ->
                 C.pixel perm ~x:x_t coord))
      | _ ->
          Err.return
            (Schedule.evaluate out_shape
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  (* Explicit int64-input promotion for [Mul_scalar]: the default arm's
     [Pointwise.Mul_scalar.Compute(S).pixel] reads through [S.load], which is
     numerically exact for this promotion ([Payload.get_float]'s I64 case is
     [Int64.to_float]) but incidental -- branch on the operand's declared
     format so the cast is the explicit [i64_to_float] step, matching
     Reshape/Permute's own precedent. Unlike those two, the output stays the
     ordinary float pixel path: [Mul_scalar]'s output format is F32
     regardless of operand format, so only the read changes, not the
     write-back. The Bool case is [Eval_direct.admit]'s concern, not this
     match's: an admitted [Mul_scalar] never has a Bool operand here. *)
  | Mul_scalar { Pointwise.Scalar_bin.x; scalar } -> (
      let x_sig = Tensor_id.Map.find x g.Graph.tensors in
      match x_sig.Tensor_sig.fmt with
      | Payload.Fmt Payload.I64 ->
          let module C = Pointwise.Mul_scalar.Compute_i64 (Direct) (Direct) in
          let x_t = Tensor_id.Map.find x operand_env in
          Err.return
            (Schedule.evaluate out_shape (fun coord ->
                 C.pixel ~scalar x_t coord))
      | _ ->
          Err.return
            (Schedule.evaluate out_shape
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  (* Dtype-preserving tensor-tensor Add/Sub/Mul: the default arm's
     [Pointwise.{Add,Sub,Mul}.Compute(S).pixel] reads both operands through
     [S.load], exact for F32 but silently lossy above 2^53 for I64 operands,
     same defect class as Reshape/Permute before their own fixes. Dispatch
     only when BOTH operands declare I64 (the only case
     [Graph_builder.{add,sub,mul}] threads an I64 output edge for, and the
     only case a real broadcasted binary op is safe to promote wholesale
     to). [Eval_direct.admit] has already rejected a Bool operand or an
     exactly-one-I64 mixed pair, so the only two domains reaching here are
     (I64, I64) and every other agreeing pair (which the default float pixel
     path already handles identically to before). *)
  | Add { Pointwise.Bin.a; b } -> (
      let a_sig = Tensor_id.Map.find a g.Graph.tensors in
      let b_sig = Tensor_id.Map.find b g.Graph.tensors in
      match (a_sig.Tensor_sig.fmt, b_sig.Tensor_sig.fmt) with
      | Payload.(Fmt I64, Fmt I64) ->
          let module C = Pointwise.Add.Compute_i64 (Direct) (Direct) in
          let a_t = Tensor_id.Map.find a operand_env in
          let b_t = Tensor_id.Map.find b operand_env in
          let a_shape = Tensor_id.Map.find a shape_env in
          let b_shape = Tensor_id.Map.find b shape_env in
          Err.return
            (Tensor.materialize_i64 out_shape (fun coord ->
                 C.pixel ~a_shape ~b_shape a_t b_t coord))
      | _ ->
          Err.return
            (Schedule.evaluate out_shape
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  | Sub { Pointwise.Bin.a; b } -> (
      let a_sig = Tensor_id.Map.find a g.Graph.tensors in
      let b_sig = Tensor_id.Map.find b g.Graph.tensors in
      match (a_sig.Tensor_sig.fmt, b_sig.Tensor_sig.fmt) with
      | Payload.(Fmt I64, Fmt I64) ->
          let module C = Pointwise.Sub.Compute_i64 (Direct) (Direct) in
          let a_t = Tensor_id.Map.find a operand_env in
          let b_t = Tensor_id.Map.find b operand_env in
          let a_shape = Tensor_id.Map.find a shape_env in
          let b_shape = Tensor_id.Map.find b shape_env in
          Err.return
            (Tensor.materialize_i64 out_shape (fun coord ->
                 C.pixel ~a_shape ~b_shape a_t b_t coord))
      | _ ->
          Err.return
            (Schedule.evaluate out_shape
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  | Mul { Pointwise.Bin.a; b } -> (
      let a_sig = Tensor_id.Map.find a g.Graph.tensors in
      let b_sig = Tensor_id.Map.find b g.Graph.tensors in
      match (a_sig.Tensor_sig.fmt, b_sig.Tensor_sig.fmt) with
      | Payload.(Fmt I64, Fmt I64) ->
          let module C = Pointwise.Mul.Compute_i64 (Direct) (Direct) in
          let a_t = Tensor_id.Map.find a operand_env in
          let b_t = Tensor_id.Map.find b operand_env in
          let a_shape = Tensor_id.Map.find a shape_env in
          let b_shape = Tensor_id.Map.find b shape_env in
          Err.return
            (Tensor.materialize_i64 out_shape (fun coord ->
                 C.pixel ~a_shape ~b_shape a_t b_t coord))
      | _ ->
          Err.return
            (Schedule.evaluate out_shape
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  (* Explicit int64-input promotion for [To_copy]'s [Float] target only --
     the EdgeNeXt/mvitv2 "I64 Arange -> Float cast" acceptance pattern's own
     promoted-consumer step. Same rationale as [Mul_scalar] above:
     [Compute(S).pixel]'s [S.load] already computes the identical value for
     this specific case ([Payload.get_float]'s I64 case is
     [Int64.to_float]), so this is architecture-only, not a value-level fix.
     [Long] is untouched -- an I64 input reaching [Long] needs no cast at all
     (I64->I64 copy), so it keeps the existing float pixel path. *)
  | To_copy { Pointwise.To_copy.target = Pointwise.To_copy.Float; x } -> (
      let x_sig = Tensor_id.Map.find x g.Graph.tensors in
      match x_sig.Tensor_sig.fmt with
      | Payload.Fmt Payload.I64 ->
          let module C = Pointwise.To_copy.Compute_i64 (Direct) (Direct) in
          let x_t = Tensor_id.Map.find x operand_env in
          Err.return
            (Schedule.evaluate out_shape (fun coord -> C.pixel x_t coord))
      | _ ->
          Err.return
            (Schedule.evaluate out_shape
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  (* The reverse direction: [To_copy]'s [Long] target on an F32 operand --
     the real "Float to I64" cast, not merely an explicit-cast architecture
     fix like the [Float] arm above. [Compute(S).pixel]'s [Long] arm is a
     bare [S.trunc], whose result [Tensor.materialize_fmt] would then read
     back as an ordinary float -- it never produces a genuine int64 payload
     cell, and never rejects NaN/infinity/out-of-range the way the design's
     policy requires. Routes through [Pointwise.To_copy.Compute_to_long],
     which raises [Err.Exn.E] via [Direct.float_to_i64] on a bad cast input,
     the same value-dependent-runtime-error convention
     [Direct.load_index]/[i64_load] already establish for a bad gather index
     or a non-I64 read -- not a new failure channel. *)
  | To_copy { Pointwise.To_copy.target = Pointwise.To_copy.Long; x } -> (
      let x_sig = Tensor_id.Map.find x g.Graph.tensors in
      match x_sig.Tensor_sig.fmt with
      | Payload.Fmt Payload.F32 ->
          let module C = Pointwise.To_copy.Compute_to_long (Direct) (Direct) in
          let x_t = Tensor_id.Map.find x operand_env in
          Err.return
            (Tensor.materialize_i64 out_shape (fun coord -> C.pixel x_t coord))
      (* An already-I64 operand needs no cast at all -- a plain identity
         copy, the same [Long]-on-I64 gap. Reachable in principle (an int64
         tensor re-asserting its own dtype), and closing it here also
         removes a real hazard: since [Graph_builder.to_copy] now declares
         this node's output I64 unconditionally, leaving this case to the
         generic `_` fallback below (which always writes via
         [Schedule.evaluate] at F32) would silently produce an F32 payload
         under a declared I64 [Tensor_sig] -- exactly the declared/actual
         format mismatch that Trim_permute and Reshape's builder each had to
         fix. *)
      | Payload.Fmt Payload.I64 ->
          let x_t = Tensor_id.Map.find x operand_env in
          Err.return
            (Tensor.materialize_i64 out_shape (fun coord ->
                 Direct.i64_load x_t coord))
      (* Design section 3's "Bool to I64 / Float: Exact 0/1 in the
         destination carrier" -- reads via [Direct.bool_load] (canonical
         true/false), not through [Payload.get_float]'s incidental float
         encoding, matching the [I64] arm's own exact-read convention
         immediately above. *)
      | Payload.Fmt Payload.Bool ->
          let x_t = Tensor_id.Map.find x operand_env in
          Err.return
            (Tensor.materialize_i64 out_shape (fun coord ->
                 if Direct.bool_load x_t coord then 1L else 0L))
      (* Every other format is unsupported (no real importer produces
         I32/F16/BF16/etc today), and [Graph_builder.to_copy] still declares
         I64 here regardless -- so this fails at checked admission, before
         any output/scratch allocation, rather than silently writing an F32
         payload under a declared I64 signature via the generic float pixel
         path below. *)
      | Payload.Fmt other ->
          Err.fail (`Unsupported_to_copy_long_source (Payload.Fmt other)))
  (* [To_copy]'s [Bool] target now writes real [Payload.Bool] storage rather
     than the F32 0.0/1.0 encoding [Compute.pixel]'s own [Bool] arm produces
     on its own -- [Graph_builder.to_copy] declares this node's output
     [Bool] unconditionally, so leaving this to the generic `_` fallback
     below (F32-only) would reproduce the exact declared/actual mismatch
     hazard the [Long] arm above already guards against. Reuses
     [Compute(Direct).pixel]'s own formula UNCHANGED (same nonzero test,
     same NaN/infinity behavior) and only changes where the result lands --
     a canonical byte via [Tensor.materialize_bool] instead of an F32 cell
     -- so this is a storage-format change, not a semantics change;
     EdgeNeXt's own mask pattern is pinned bit-for-bit by
     [bool_acceptance_test.ml] against exactly this formula. Every other
     operand format is rejected at checked admission, matching the [Long]
     arm's own convention -- no real corpus caller casts a non-F32 operand
     to [Bool] today. *)
  | To_copy { Pointwise.To_copy.target = Pointwise.To_copy.Bool; x } -> (
      let x_sig = Tensor_id.Map.find x g.Graph.tensors in
      match x_sig.Tensor_sig.fmt with
      | Payload.Fmt Payload.F32 ->
          let module C = Pointwise.To_copy.Compute (Direct) in
          let x_t = Tensor_id.Map.find x operand_env in
          Err.return
            (Tensor.materialize_bool out_shape (fun coord ->
                 C.pixel Pointwise.To_copy.Bool x_t coord <> 0.0))
      (* I64 to Bool is an exact comparison with 0L -- an exact int64 zero
         test, not a route through [Payload.get_float]/[Compute.pixel]'s own
         float nonzero test (which would still happen to agree for every
         representable int64, since [Int64.to_float 0L = 0.0] and every
         nonzero int64 has a nonzero float image, but reading through float
         is the exact "integer-to-float, not an explicit expression cast"
         hazard, regardless of numerical agreement). *)
      | Payload.Fmt Payload.I64 ->
          let x_t = Tensor_id.Map.find x operand_env in
          Err.return
            (Tensor.materialize_bool out_shape (fun coord ->
                 not (Int64.equal (Direct.i64_load x_t coord) 0L)))
      | Payload.Fmt other ->
          Err.fail (`Unsupported_to_copy_bool_source (Payload.Fmt other)))
  (* [Bitwise_not] now writes real [Payload.Bool] storage too, matching
     [Graph_builder.bitwise_not]'s own unconditional [Bool] output
     declaration -- real ATen's [bitwise_not] on a bool operand produces a
     bool result, and nothing routes an integer operand here today (see
     [Pointwise.Bitwise_not]'s own comment). No operand-format branch is
     needed, unlike [To_copy]'s casts: [Compute(Direct).pixel]'s existing
     formula already reads ANY operand format through
     [S.load]/[Payload.get_float] (format-agnostic), so this arm only
     changes where the result lands. *)
  | Bitwise_not { Pointwise.Bitwise_not.x } ->
      let module C = Pointwise.Bitwise_not.Compute (Direct) in
      let x_t = Tensor_id.Map.find x operand_env in
      Err.return
        (Tensor.materialize_bool out_shape (fun coord ->
             C.pixel x_t coord <> 0.0))
  (* [Eq_scalar] mirrors [Gt_scalar]'s own split exactly (using
     [SEMANTICS.eq] instead of [S.lt]/[S.select]): [Compute]'s formula is
     [SEMANTICS]-generic (shared with [Symbolic] via [Eval_op.Make], which
     still writes a plain float 0./1.), and only [Eval_direct] intercepts it
     to land genuine [Payload.Bool] storage, matching
     [Graph_builder.eq_scalar]'s own unconditional [Bool] output
     declaration. *)
  | Eq_scalar { Pointwise.Scalar_bin.x; scalar } ->
      let module C = Pointwise.Eq_scalar.Compute (Direct) in
      let x_t = Tensor_id.Map.find x operand_env in
      Err.return
        (Tensor.materialize_bool out_shape (fun coord ->
             C.pixel ~scalar x_t coord <> 0.0))
  (* [Eq_tensor] mirrors [Eq_scalar]'s own split, broadcast via [Binary]
     instead of [Scalar_binary] since both operands are runtime tensors:
     [Compute]'s formula is [SEMANTICS]-generic (shared with [Symbolic]),
     and only [Eval_direct] intercepts it to land genuine [Payload.Bool]
     storage, matching [Graph_builder.eq_tensor]'s own unconditional [Bool]
     output declaration. *)
  | Eq_tensor { Pointwise.Bin.a; b } ->
      let module C = Pointwise.Eq_tensor.Compute (Direct) in
      let a_t = Tensor_id.Map.find a operand_env in
      let b_t = Tensor_id.Map.find b operand_env in
      let a_shape = Tensor_id.Map.find a shape_env in
      let b_shape = Tensor_id.Map.find b shape_env in
      Err.return
        (Tensor.materialize_bool out_shape (fun coord ->
             C.pixel ~a_shape ~b_shape a_t b_t coord <> 0.0))
  (* [Gt_scalar] mirrors [Bitwise_not]'s own split: [Compute]'s formula is
     [SEMANTICS]-generic (shared with [Symbolic] via [Eval_op.Make], which
     still writes a plain float 0./1.), and only [Eval_direct] intercepts it
     to land genuine [Payload.Bool] storage, matching
     [Graph_builder.gt_scalar]'s own unconditional [Bool] output
     declaration. *)
  | Gt_scalar { Pointwise.Scalar_bin.x; scalar } ->
      let module C = Pointwise.Gt_scalar.Compute (Direct) in
      let x_t = Tensor_id.Map.find x operand_env in
      Err.return
        (Tensor.materialize_bool out_shape (fun coord ->
             C.pixel ~scalar x_t coord <> 0.0))
  (* [Ne_scalar] mirrors [Eq_scalar]'s own split exactly (negated):
     [Compute]'s formula is [SEMANTICS]-generic (shared with [Symbolic] via
     [Eval_op.Make], which still writes a plain float 0./1.), and only
     [Eval_direct] intercepts it to land genuine [Payload.Bool] storage,
     matching [Graph_builder.ne_scalar]'s own unconditional [Bool] output
     declaration. *)
  | Ne_scalar { Pointwise.Scalar_bin.x; scalar } ->
      let module C = Pointwise.Ne_scalar.Compute (Direct) in
      let x_t = Tensor_id.Map.find x operand_env in
      Err.return
        (Tensor.materialize_bool out_shape (fun coord ->
             C.pixel ~scalar x_t coord <> 0.0))
  (* [Ne_tensor] mirrors [Eq_tensor]'s own split exactly (negated), matching
     [Graph_builder.ne_tensor]'s own unconditional [Bool] output
     declaration. *)
  | Ne_tensor { Pointwise.Bin.a; b } ->
      let module C = Pointwise.Ne_tensor.Compute (Direct) in
      let a_t = Tensor_id.Map.find a operand_env in
      let b_t = Tensor_id.Map.find b operand_env in
      let a_shape = Tensor_id.Map.find a shape_env in
      let b_shape = Tensor_id.Map.find b shape_env in
      Err.return
        (Tensor.materialize_bool out_shape (fun coord ->
             C.pixel ~a_shape ~b_shape a_t b_t coord <> 0.0))
  (* The index output of [Max_pool2d_with_indices], its adaptive twin and
     [Max_dim] is declared I64 by [Graph_builder] (ATen returns int64
     indices). [index_pixel] carries the flat index in a double -- a small
     non-negative integer, so the conversion to int64 is exact -- and only
     the landing storage changes, not the tie or NaN policy. The value
     output falls through to the generic F32 path. *)
  | Max_pool2d_with_indices { Pool.MaxPool2dWithIndices.params; x }
    when Output_ordinal.equal output Output_ordinal.one ->
      let module C = Pool.MaxPool2dWithIndices.Compute (Direct) in
      Err.return
        (index_i64 out_shape
           ~x_shape:(Tensor_id.Map.find x shape_env)
           ~x:(Tensor_id.Map.find x operand_env)
           (C.index_pixel params))
  | Adaptive_max_pool2d_with_indices
      { Pool.AdaptiveMaxPool2dWithIndices.params; x }
    when Output_ordinal.equal output Output_ordinal.one ->
      let module C = Pool.AdaptiveMaxPool2dWithIndices.Compute (Direct) in
      Err.return
        (index_i64 out_shape
           ~x_shape:(Tensor_id.Map.find x shape_env)
           ~x:(Tensor_id.Map.find x operand_env)
           (C.index_pixel params))
  | Max_dim { Reduce.MaxDim.params; x }
    when Output_ordinal.equal output Output_ordinal.one ->
      let module C = Reduce.MaxDim.Compute (Direct) in
      Err.return
        (index_i64 out_shape
           ~x_shape:(Tensor_id.Map.find x shape_env)
           ~x:(Tensor_id.Map.find x operand_env)
           (C.index_pixel params))
  | _ ->
      Err.return
        (Schedule.evaluate out_shape
           (E.pixel op ~output
              ~operand:(fun r -> Tensor_id.Map.find r operand_env)
              ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
              ~fill))
