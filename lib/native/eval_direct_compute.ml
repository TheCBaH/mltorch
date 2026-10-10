(* See eval_direct_compute.mli. *)

open Graph_ir
module E = Eval_op.Make (Direct)

type error =
  [ `Arange_i64_overflow of Factory.Arange.Overflow.t
  | `Embedding_index_out_of_range of Embedding.Embedding.Index_out_of_range.t
  | Tensor.dst_error
  | `Unsupported_to_copy_bool_source of Payload.packed_fmt
  | `Unsupported_to_copy_long_source of Payload.packed_fmt ]

let pp_error ppf : [< error ] -> unit = function
  | `Arange_i64_overflow { Factory.Arange.Overflow.start; step; i } ->
      Format.fprintf ppf
        "arange: exact int64 generation overflows at start=%Ld step=%Ld i=%d"
        start step i
  | `Embedding_index_out_of_range e ->
      Embedding.Embedding.Index_out_of_range.pp ppf e
  | #Tensor.dst_error as e -> Tensor.pp_dst_error ppf e
  | `Unsupported_to_copy_bool_source (Payload.Fmt f) ->
      Format.fprintf ppf
        "to_copy: Bool target has no exact Bool output for a %s source"
        (Payload.fmt_name f)
  | `Unsupported_to_copy_long_source (Payload.Fmt f) ->
      Format.fprintf ppf
        "to_copy: Long target has no exact I64 output for a %s source"
        (Payload.fmt_name f)

(* An index-style output landed as exact int64 storage. *)
let index_i64 dst ~x_shape ~x pixel =
  Tensor.write_i64 dst (fun coord -> Int64.of_float (pixel ~x_shape ~x coord))

(* The destination is the result: an arm writes into it and hands it back. *)
let finish dst r = Err.map (fun () -> dst) r

let compute_arms (g : graph) (op : op) ~(output : Output_ordinal.t) ~out_shape
    ~dst ~operand_env ~shape_env ~fill : (Tensor.packed, [> error ]) Err.t =
  match op with
  | Unbind { Split.Unbind.params; x } ->
      finish dst
        (Tensor.unbind_into
           (Tensor_id.Map.find x operand_env)
           dst ~axis:params.axis ~output ~shape:out_shape)
  (* Same dtype-preserving bypass as [Unbind], and for the same reason:
     [offset] is the sum of every earlier piece's size, computed the same way
     [Eval_op]'s arm computes it for the generic path. *)
  | Split_with_sizes { Split.Split_with_sizes.params; x } ->
      let offset =
        Split.Split_with_sizes.offset_of ~output
          params.Split.Split_with_sizes.sizes
      in
      finish dst
        (Tensor.split_with_sizes_into
           (Tensor_id.Map.find x operand_env)
           dst ~axis:params.Split.Split_with_sizes.axis ~offset ~shape:out_shape)
  | Zeros { Factory.Zeros.params = _ } ->
      finish dst (Tensor.write_float dst (fun _ -> 0.))
  | New_ones { Factory.New_ones.params } -> (
      match params.fmt with
      | Payload.Fmt Payload.Bool ->
          finish dst (Tensor.write_bool dst (fun _ -> true))
      | _ -> finish dst (Tensor.write_float dst (fun _ -> 1.)))
  | Eye { Factory.Eye.params = _ } ->
      finish dst
        (Tensor.write_float dst (fun coord ->
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
              finish dst
                (Err.bind
                   (Err.Escape.with_escape (fun esc ->
                        Tensor.write_i64 dst (fun coord ->
                            Err.Escape.or_throw esc
                              (Factory.Arange.value_i64_exact e
                                 (Dim.to_int coord.Vec6.c)))))
                   Fun.id)
          | None ->
              finish dst
                (Tensor.write_i64 dst (fun coord ->
                     Int64.of_float
                       (Factory.Arange.value params (Dim.to_int coord.Vec6.c))))
          )
      | _ ->
          finish dst
            (Tensor.write_float dst (fun coord ->
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
          finish dst
            (Tensor.write_i64 dst (fun coord ->
                 C.pixel params ~x_shape ~x:x_t coord))
      | _ ->
          finish dst
            (Schedule.evaluate_into dst
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
          finish dst
            (Tensor.write_i64 dst (fun coord -> C.pixel perm ~x:x_t coord))
      | _ ->
          finish dst
            (Schedule.evaluate_into dst
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  (* A gather from a bool [self] lands as genuine bool storage (the builder
     keeps [self]'s format); the float pixel path cannot write a bool
     destination. *)
  | Index_pair { Index_tensor.Index_pair.params; self; index0; index1 } -> (
      let (Tensor.Tensor d) = dst in
      match d.Tensor.payload.Payload.fmt with
      | Payload.Bool ->
          let module C = Index_tensor.Index_pair.Compute (Direct) in
          let t r = Tensor_id.Map.find r operand_env in
          let shape r = Tensor_id.Map.find r shape_env in
          finish dst
            (Tensor.write_bool dst (fun coord ->
                 C.pixel params ~self_shape:(shape self)
                   ~index0_shape:(shape index0) ~index1_shape:(shape index1)
                   ~self:(t self) ~index0:(t index0) ~index1:(t index1) coord
                 <> 0.0))
      | _ ->
          finish dst
            (Schedule.evaluate_into dst
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  (* [Weight_norm]: the norm over every axis but [axis] is computed once per
     position along it, not once per element -- the generic pixel re-sums the
     whole tensor for each output and is only the definition. Squares are
     accumulated in binary64 and the quotient taken per element, as that pixel
     does. *)
  | Weight_norm { Weight_norm.Weight_norm.params; v; g } ->
      let v_t = Tensor_id.Map.find v operand_env in
      let g_t = Tensor_id.Map.find g operand_env in
      let v_shape = Tensor_id.Map.find v shape_env in
      let axis = params.Weight_norm.Weight_norm.axis in
      let extent = (Vec6.get v_shape axis :> int) in
      let sums = Array.make extent 0. in
      Vec6.iter v_shape (fun c ->
          let k = (Vec6.get c axis :> int) in
          let x = Direct.load v_t c in
          sums.(k) <- sums.(k) +. (x *. x));
      finish dst
        (Tensor.write_float dst (fun coord ->
             let k = (Vec6.get coord axis :> int) in
             let g_at =
               Direct.load g_t
                 (Vec6.mapi
                    (fun a i ->
                      if Axis.equal a axis then i else Direct.index_zero)
                    coord)
             in
             Direct.load v_t coord *. (g_at /. sqrt sums.(k))))
  (* [Full_like] writes its fill in [x]'s own format; an int64 fill must be a
     whole number, never silently truncated. *)
  | Full_like { Pointwise.Full_like.value; _ } -> (
      let (Tensor.Tensor d) = dst in
      match d.Tensor.payload.Payload.fmt with
      | Payload.I64 ->
          if not (Float.is_integer value) then
            Err.or_raise
              ~pp_error:(fun fmt v ->
                Fmt.pf fmt
                  "full_like: %g is not a whole number for an int64 tensor" v)
              (Err.fail value)
          else finish dst (Tensor.write_i64 dst (fun _ -> Int64.of_float value))
      | Payload.Bool ->
          finish dst (Tensor.write_bool dst (fun _ -> value <> 0.))
      | _ -> finish dst (Tensor.write_float dst (fun _ -> value)))
  (* Int64 operands are compared exactly, never through the float domain. *)
  | Min_other { Pointwise.Bin.a; b } -> (
      let a_sig = Tensor_id.Map.find a g.Graph.tensors in
      match a_sig.Tensor_sig.fmt with
      | Payload.Fmt Payload.I64 ->
          let module B = Pointwise_binary.Binary_i64 (Direct) (Direct) in
          let a_t = Tensor_id.Map.find a operand_env in
          let b_t = Tensor_id.Map.find b operand_env in
          let a_shape = Tensor_id.Map.find a shape_env in
          let b_shape = Tensor_id.Map.find b shape_env in
          finish dst
            (Tensor.write_i64 dst (fun coord ->
                 B.pixel
                   ~combine:(fun x y -> if Direct.i64_lt y x then y else x)
                   ~a_shape ~b_shape a_t b_t coord))
      | _ ->
          finish dst
            (Schedule.evaluate_into dst
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  | Where_self { Pointwise.Where_self.condition; x; y } -> (
      let x_sig = Tensor_id.Map.find x g.Graph.tensors in
      match x_sig.Tensor_sig.fmt with
      | Payload.Fmt Payload.I64 ->
          let c_t = Tensor_id.Map.find condition operand_env in
          let x_t = Tensor_id.Map.find x operand_env in
          let y_t = Tensor_id.Map.find y operand_env in
          let at r t coord =
            Pointwise.broadcast_coord ~index_zero:Direct.index_zero
              (Tensor_id.Map.find r shape_env)
              coord
            |> fun c -> t c
          in
          finish dst
            (Tensor.write_i64 dst (fun coord ->
                 let cond = at condition (Direct.load c_t) coord in
                 if cond = 0. then at y (Direct.i64_load y_t) coord
                 else at x (Direct.i64_load x_t) coord))
      | _ ->
          finish dst
            (Schedule.evaluate_into dst
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  (* An I64 operand plus an integral scalar stays int64 (the builder threads
     an I64 output edge exactly then): the exact add, never the float domain.
     Every other case keeps the float pixel path. *)
  | Add_scalar { Pointwise.Scalar_bin.x; scalar } -> (
      let (Tensor.Tensor d) = dst in
      match d.Tensor.payload.Payload.fmt with
      | Payload.I64 ->
          let module T = struct
            type 'a repr = 'a

            let i64_load = Direct.i64_load
            let i64_binary = Direct.i64_binary
            let typed_const = Direct.typed_const
          end in
          let module C = Pointwise.Add_scalar.Compute_i64 (Direct) (T) in
          let x_t = Tensor_id.Map.find x operand_env in
          finish dst
            (Tensor.write_i64 dst (fun coord -> C.pixel ~scalar x_t coord))
      | _ ->
          finish dst
            (Schedule.evaluate_into dst
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  (* Data movement over an int64 operand copies exactly: [Clone], [Expand],
     [Repeat] and [RepeatInterleave] read through [Compute_i64], as [Reshape]
     and [Slice] do. Any other format keeps the float pixel path. *)
  | Clone { Pointwise.Clone.x } -> (
      let x_sig = Tensor_id.Map.find x g.Graph.tensors in
      match x_sig.Tensor_sig.fmt with
      | Payload.Fmt Payload.I64 ->
          let module C = Pointwise.Clone.Compute_i64 (Direct) (Direct) in
          let x_t = Tensor_id.Map.find x operand_env in
          finish dst (Tensor.write_i64 dst (fun coord -> C.pixel x_t coord))
      | _ ->
          finish dst
            (Schedule.evaluate_into dst
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  | Expand { Pointwise.Expand.params = _; x } -> (
      let x_sig = Tensor_id.Map.find x g.Graph.tensors in
      match x_sig.Tensor_sig.fmt with
      | Payload.Fmt Payload.I64 ->
          let module C = Pointwise.Expand.Compute_i64 (Direct) (Direct) in
          let x_t = Tensor_id.Map.find x operand_env in
          let x_shape = Tensor_id.Map.find x shape_env in
          finish dst
            (Tensor.write_i64 dst (fun coord -> C.pixel ~x_shape x_t coord))
      | _ ->
          finish dst
            (Schedule.evaluate_into dst
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  | Repeat { Repeat.Repeat.params = _; x } -> (
      let x_sig = Tensor_id.Map.find x g.Graph.tensors in
      match x_sig.Tensor_sig.fmt with
      | Payload.Fmt Payload.I64 ->
          let module C = Repeat.Repeat.Compute_i64 (Direct) (Direct) in
          let x_t = Tensor_id.Map.find x operand_env in
          let x_shape = Tensor_id.Map.find x shape_env in
          finish dst
            (Tensor.write_i64 dst (fun coord -> C.pixel ~x_shape x_t coord))
      | _ ->
          finish dst
            (Schedule.evaluate_into dst
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  | RepeatInterleave { Repeat.RepeatInterleave.params; x } -> (
      let x_sig = Tensor_id.Map.find x g.Graph.tensors in
      match x_sig.Tensor_sig.fmt with
      | Payload.Fmt Payload.I64 ->
          let module C = Repeat.RepeatInterleave.Compute_i64 (Direct) (Direct)
          in
          let x_t = Tensor_id.Map.find x operand_env in
          finish dst
            (Tensor.write_i64 dst (fun coord -> C.pixel params x_t coord))
      | _ ->
          finish dst
            (Schedule.evaluate_into dst
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  (* Dtype-preserving Slice, the same shape as Permute above: an I64 source is
     sliced through [Compute_i64]/[i64_load], exact beyond 2^53; every other
     format keeps the float pixel path. *)
  | Slice { Split.Slice.params; x } -> (
      let x_sig = Tensor_id.Map.find x g.Graph.tensors in
      match x_sig.Tensor_sig.fmt with
      | Payload.Fmt Payload.I64 ->
          let module C = Split.Slice.Compute_i64 (Direct) (Direct) in
          let x_t = Tensor_id.Map.find x operand_env in
          finish dst
            (Tensor.write_i64 dst (fun coord -> C.pixel params ~x:x_t coord))
      | _ ->
          finish dst
            (Schedule.evaluate_into dst
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
      let (Tensor.Tensor d) = dst in
      match (d.Tensor.payload.Payload.fmt, x_sig.Tensor_sig.fmt) with
      (* The builder threads an I64 output edge only for an I64 operand and an
         integral scalar: that is the exact integer product. *)
      | Payload.I64, _ ->
          let module T = struct
            type 'a repr = 'a

            let i64_load = Direct.i64_load
            let i64_binary = Direct.i64_binary
            let typed_const = Direct.typed_const
          end in
          let module C = Pointwise.Mul_scalar.Compute_i64_exact (Direct) (T) in
          let x_t = Tensor_id.Map.find x operand_env in
          finish dst
            (Tensor.write_i64 dst (fun coord -> C.pixel ~scalar x_t coord))
      | _, Payload.Fmt Payload.I64 ->
          let module C = Pointwise.Mul_scalar.Compute_i64 (Direct) (Direct) in
          let x_t = Tensor_id.Map.find x operand_env in
          finish dst
            (Schedule.evaluate_into dst (fun coord -> C.pixel ~scalar x_t coord))
      | _ ->
          finish dst
            (Schedule.evaluate_into dst
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
  (* Dtype-preserving [Abs]: an I64 operand takes the exact int64 pixel (read
     through [i64_load], negated in wrapping int64 arithmetic), never the float
     pixel, whose [S.load] is lossy above 2^53. The builder threads the I64
     output edge for exactly this case, so the destination is an I64 tensor. *)
  (* ATen's embedding is [index_select]: every index must lie in [0, V). The
     gather this delegates to also takes [-V, -1] (it wraps, as advanced
     indexing does), so the strict rule is enforced here, over every index
     element before any row is read. A bad index is an error row, never a
     clamped or wrapped read. *)
  | Embedding { Embedding.Embedding.weight; indices; _ } -> (
      let vocabulary = Vec6.get (Tensor_id.Map.find weight shape_env) Axis.W in
      let indices_t = Tensor_id.Map.find indices operand_env in
      let bad = ref None in
      Vec6.iter (Tensor_id.Map.find indices shape_env) (fun coord ->
          if !bad = None then
            let raw =
              Err.or_raise
                ~pp_error:(fun fmt (`Wrong_format (Payload.Fmt f)) ->
                  Fmt.pf fmt "embedding indices must be I64, got %s"
                    (Payload.fmt_name f))
                (Tensor.read_i64_at6 indices_t (fun a ->
                     (Vec6.get coord a :> int)))
            in
            if
              Int64.compare raw 0L < 0
              || Int64.compare raw (Int64.of_int (vocabulary :> int)) >= 0
            then bad := Some raw);
      match !bad with
      | Some raw ->
          Err.fail
            (`Embedding_index_out_of_range
               { Embedding.Embedding.Index_out_of_range.raw; vocabulary })
      | None ->
          finish dst
            (Schedule.evaluate_into dst
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  | Abs { Pointwise.Abs.x } -> (
      let x_sig = Tensor_id.Map.find x g.Graph.tensors in
      match x_sig.Tensor_sig.fmt with
      | Payload.Fmt Payload.I64 ->
          (* [Direct]'s own [b] is the abstract predicate of [SEMANTICS]; the
             typed section's selects and comparisons are on [bool]. *)
          let module T = struct
            type 'a repr = 'a
            type b = bool

            let i64_load = Direct.i64_load
            let i64_binary = Direct.i64_binary
            let i64_lt = Direct.i64_lt
            let typed_const = Direct.typed_const
            let typed_select = Direct.typed_select
          end in
          let module C = Pointwise.Abs.Compute_i64 (Direct) (T) in
          let x_t = Tensor_id.Map.find x operand_env in
          finish dst (Tensor.write_i64 dst (fun coord -> C.pixel x_t coord))
      | _ ->
          finish dst
            (Schedule.evaluate_into dst
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
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
          finish dst
            (Tensor.write_i64 dst (fun coord ->
                 C.pixel ~a_shape ~b_shape a_t b_t coord))
      | _ ->
          finish dst
            (Schedule.evaluate_into dst
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
          finish dst
            (Tensor.write_i64 dst (fun coord ->
                 C.pixel ~a_shape ~b_shape a_t b_t coord))
      | _ ->
          finish dst
            (Schedule.evaluate_into dst
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
          finish dst
            (Tensor.write_i64 dst (fun coord ->
                 C.pixel ~a_shape ~b_shape a_t b_t coord))
      | _ ->
          finish dst
            (Schedule.evaluate_into dst
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
          finish dst
            (Schedule.evaluate_into dst (fun coord -> C.pixel x_t coord))
      | _ ->
          finish dst
            (Schedule.evaluate_into dst
               (E.pixel op ~output
                  ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                  ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                  ~fill)))
  (* [To_copy]'s [Int] target (int32): the value is carried in an int64 cell,
     since the engine has no 32-bit integer edge, and a value outside the int32
     range is an error raised through the same [Err.or_raise] boundary a bad
     [Long] cast uses -- never the wrap-around ATen's [static_cast] gives, which
     this carrier could not reproduce. The Symbolic route does not range-check:
     it is the same carrier without the guard, so an out-of-range value there is
     carried, not wrapped. *)
  | To_copy { Pointwise.To_copy.target = Pointwise.To_copy.Int; x } -> (
      let x_sig = Tensor_id.Map.find x g.Graph.tensors in
      let x_t = Tensor_id.Map.find x operand_env in
      let in_range v =
        if Int64.compare v (-2147483648L) < 0 || Int64.compare v 2147483647L > 0
        then
          Err.or_raise
            ~pp_error:(fun fmt v ->
              Fmt.pf fmt "to_copy int32: %Ld is outside [-2^31, 2^31)" v)
            (Err.fail v)
        else v
      in
      match x_sig.Tensor_sig.fmt with
      | Payload.Fmt Payload.I64 ->
          finish dst
            (Tensor.write_i64 dst (fun coord ->
                 in_range (Direct.i64_load x_t coord)))
      | Payload.Fmt Payload.Bool ->
          finish dst
            (Tensor.write_i64 dst (fun coord ->
                 if Direct.bool_load x_t coord then 1L else 0L))
      | Payload.Fmt Payload.F32 ->
          let module C = Pointwise.To_copy.Compute_to_long (Direct) (Direct) in
          finish dst
            (Tensor.write_i64 dst (fun coord -> in_range (C.pixel x_t coord)))
      | Payload.Fmt other ->
          Err.fail (`Unsupported_to_copy_long_source (Payload.Fmt other)))
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
          finish dst (Tensor.write_i64 dst (fun coord -> C.pixel x_t coord))
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
          finish dst
            (Tensor.write_i64 dst (fun coord -> Direct.i64_load x_t coord))
      (* Design section 3's "Bool to I64 / Float: Exact 0/1 in the
         destination carrier" -- reads via [Direct.bool_load] (canonical
         true/false), not through [Payload.get_float]'s incidental float
         encoding, matching the [I64] arm's own exact-read convention
         immediately above. *)
      | Payload.Fmt Payload.Bool ->
          let x_t = Tensor_id.Map.find x operand_env in
          finish dst
            (Tensor.write_i64 dst (fun coord ->
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
          finish dst
            (Tensor.write_bool dst (fun coord ->
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
          finish dst
            (Tensor.write_bool dst (fun coord ->
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
      finish dst (Tensor.write_bool dst (fun coord -> C.pixel x_t coord <> 0.0))
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
      finish dst
        (Tensor.write_bool dst (fun coord -> C.pixel ~scalar x_t coord <> 0.0))
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
      finish dst
        (Tensor.write_bool dst (fun coord ->
             C.pixel ~a_shape ~b_shape a_t b_t coord <> 0.0))
  (* [Bitwise_and], [Ge_scalar], [Le_tensor] and [Lt_scalar] mirror
     [Gt_scalar]/[Eq_tensor]'s split below: the [SEMANTICS]-generic formula
     yields 0./1., and only [Eval_direct] lands it as genuine [Bool] storage,
     matching the builder's unconditional [Bool] output declaration. *)
  | Bitwise_and { Pointwise.Bin.a; b } ->
      let module C = Pointwise.Bitwise_and.Compute (Direct) in
      let a_t = Tensor_id.Map.find a operand_env in
      let b_t = Tensor_id.Map.find b operand_env in
      let a_shape = Tensor_id.Map.find a shape_env in
      let b_shape = Tensor_id.Map.find b shape_env in
      finish dst
        (Tensor.write_bool dst (fun coord ->
             C.pixel ~a_shape ~b_shape a_t b_t coord <> 0.0))
  | Ge_scalar { Pointwise.Scalar_bin.x; scalar } ->
      let module C = Pointwise.Ge_scalar.Compute (Direct) in
      let x_t = Tensor_id.Map.find x operand_env in
      finish dst
        (Tensor.write_bool dst (fun coord -> C.pixel ~scalar x_t coord <> 0.0))
  | Le_tensor { Pointwise.Bin.a; b } ->
      let module C = Pointwise.Le_tensor.Compute (Direct) in
      let a_t = Tensor_id.Map.find a operand_env in
      let b_t = Tensor_id.Map.find b operand_env in
      let a_shape = Tensor_id.Map.find a shape_env in
      let b_shape = Tensor_id.Map.find b shape_env in
      finish dst
        (Tensor.write_bool dst (fun coord ->
             C.pixel ~a_shape ~b_shape a_t b_t coord <> 0.0))
  | Lt_scalar { Pointwise.Scalar_bin.x; scalar } ->
      let module C = Pointwise.Lt_scalar.Compute (Direct) in
      let x_t = Tensor_id.Map.find x operand_env in
      finish dst
        (Tensor.write_bool dst (fun coord -> C.pixel ~scalar x_t coord <> 0.0))
  (* [Gt_scalar] mirrors [Bitwise_not]'s own split: [Compute]'s formula is
     [SEMANTICS]-generic (shared with [Symbolic] via [Eval_op.Make], which
     still writes a plain float 0./1.), and only [Eval_direct] intercepts it
     to land genuine [Payload.Bool] storage, matching
     [Graph_builder.gt_scalar]'s own unconditional [Bool] output
     declaration. *)
  | Gt_scalar { Pointwise.Scalar_bin.x; scalar } ->
      let module C = Pointwise.Gt_scalar.Compute (Direct) in
      let x_t = Tensor_id.Map.find x operand_env in
      finish dst
        (Tensor.write_bool dst (fun coord -> C.pixel ~scalar x_t coord <> 0.0))
  (* [Ne_scalar] mirrors [Eq_scalar]'s own split exactly (negated):
     [Compute]'s formula is [SEMANTICS]-generic (shared with [Symbolic] via
     [Eval_op.Make], which still writes a plain float 0./1.), and only
     [Eval_direct] intercepts it to land genuine [Payload.Bool] storage,
     matching [Graph_builder.ne_scalar]'s own unconditional [Bool] output
     declaration. *)
  | Ne_scalar { Pointwise.Scalar_bin.x; scalar } ->
      let module C = Pointwise.Ne_scalar.Compute (Direct) in
      let x_t = Tensor_id.Map.find x operand_env in
      finish dst
        (Tensor.write_bool dst (fun coord -> C.pixel ~scalar x_t coord <> 0.0))
  (* [Ne_tensor] mirrors [Eq_tensor]'s own split exactly (negated), matching
     [Graph_builder.ne_tensor]'s own unconditional [Bool] output
     declaration. *)
  | Ne_tensor { Pointwise.Bin.a; b } ->
      let module C = Pointwise.Ne_tensor.Compute (Direct) in
      let a_t = Tensor_id.Map.find a operand_env in
      let b_t = Tensor_id.Map.find b operand_env in
      let a_shape = Tensor_id.Map.find a shape_env in
      let b_shape = Tensor_id.Map.find b shape_env in
      finish dst
        (Tensor.write_bool dst (fun coord ->
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
      finish dst
        (index_i64 dst
           ~x_shape:(Tensor_id.Map.find x shape_env)
           ~x:(Tensor_id.Map.find x operand_env)
           (C.index_pixel params))
  | Adaptive_max_pool2d_with_indices
      { Pool.AdaptiveMaxPool2dWithIndices.params; x }
    when Output_ordinal.equal output Output_ordinal.one ->
      let module C = Pool.AdaptiveMaxPool2dWithIndices.Compute (Direct) in
      finish dst
        (index_i64 dst
           ~x_shape:(Tensor_id.Map.find x shape_env)
           ~x:(Tensor_id.Map.find x operand_env)
           (C.index_pixel params))
  | Argmax { Reduce.Argmax.params; x } ->
      let module C = Reduce.Argmax.Compute (Direct) in
      finish dst
        (index_i64 dst
           ~x_shape:(Tensor_id.Map.find x shape_env)
           ~x:(Tensor_id.Map.find x operand_env)
           (C.pixel params))
  | Max_dim { Reduce.MaxDim.params; x }
    when Output_ordinal.equal output Output_ordinal.one ->
      let module C = Reduce.MaxDim.Compute (Direct) in
      finish dst
        (index_i64 dst
           ~x_shape:(Tensor_id.Map.find x shape_env)
           ~x:(Tensor_id.Map.find x operand_env)
           (C.index_pixel params))
  | _ ->
      finish dst
        (Schedule.evaluate_into dst
           (E.pixel op ~output
              ~operand:(fun r -> Tensor_id.Map.find r operand_env)
              ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
              ~fill))

let compute g op ~output ~out_shape ~dst ~operand_env ~shape_env ~fill =
  let open Err.Syntax in
  let* () = Tensor.check_shape dst out_shape in
  compute_arms g op ~output ~out_shape ~dst ~operand_env ~shape_env ~fill
