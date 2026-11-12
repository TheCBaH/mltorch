(* Copy, dtype-conversion and identity family (clone/_to_copy/type_as/copy/alias/detach) op-dispatch arms for [Op_bridge], split from op_bridge_shape.ml. See
   op_bridge.ml for the [dispatch] facade that folds every family module
   (including this one) into one lookup. *)

open Pytorch_types
open Op_bridge_error
open Op_bridge_decode
module D = Interp_decode

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list, error) Err.t
    option =
  match node.target with
  | "torch.ops.aten.clone.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         (* Native's engine has exactly one physical layout per shape --
            there is no channels-last stride concept anywhere in the six-axis
            frame -- so a request to make the result CONTIGUOUS, or to
            PRESERVE whatever format the input already has, is always already
            true: every native tensor is already in that one dense layout.
            (Confirmed against the corpus: every `clone.default` occurrence
            across the 100-model sweep requests either no format or exactly
            `ContiguousFormat` -- see `.ai/pt2_model_support.md`.) A request
            for an actual different physical arrangement (channels-last) asks
            for something this engine cannot represent, so it is refused
            rather than silently treated as a no-op. *)
         let* mf =
           decode_result (D.memory_format_opt_arg_result node "memory_format")
         in
         match mf with
         | None | Some (Aten_memory_format.Contiguous | Preserve) ->
             build_g ~name:"clone" [ x ] (function
               | [ x_id ] ->
                   let open Graph_builder in
                   let+ y = clone x_id in
                   [ y ]
               | operands -> arity_mismatch ~op:node.Node.target operands)
         | Some Aten_memory_format.ChannelsLast ->
             fail (`Unsupported_memory_format `Channels_last)
         | Some Aten_memory_format.ChannelsLast3d ->
             fail (`Unsupported_memory_format `Channels_last_3d))
  (* `_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None,
     Device? device=None, bool? pin_memory=None, bool non_blocking=False,
     MemoryFormat? memory_format=None) -> Tensor`. Restricted to the
     three-way [Pointwise.To_copy.target] domain the corpus actually uses
     (bool/float/long) -- see that module's own comment for why there is no
     Native value representation to target outside it. [dtype=None] means
     "keep self's own dtype", which this engine cannot distinguish from the
     float target (there is no per-tensor dtype tag on an ordinary compute
     edge -- see graph_builder.ml's "op-output edges are F32" note), so it
     is treated as [Float], the identity. [non_blocking] is a
     device-transfer hint, read-and-discarded the same way it is in
     [copy.default]'s arm above; [layout]/[device]/[pin_memory] are rejected
     unless omitted/default, the same [zeros.default] precedent;
     [memory_format] is rejected the same way [clone.default]'s own arm
     rejects a real relayout request. *)
  | "torch.ops.aten._to_copy.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         let* target =
           match D.find_arg node "dtype" with
           | None | Some (Argument.None _) -> return Pointwise.To_copy.Float
           | Some (Argument.Scalar_type Pytorch_types.ScalarType.BOOL) ->
               return Pointwise.To_copy.Bool
           | Some (Argument.Scalar_type Pytorch_types.ScalarType.FLOAT) ->
               return Pointwise.To_copy.Float
           | Some (Argument.Scalar_type Pytorch_types.ScalarType.LONG) ->
               return Pointwise.To_copy.Long
           | Some _ ->
               fail
                 (`Validation_failure
                    "_to_copy: only default/BOOL/FLOAT/LONG dtype are supported")
         in
         let reject_absent name =
           match D.find_arg node name with
           | None | Some (Argument.None _) -> return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    ("_to_copy: " ^ name ^ " must be omitted or None"))
         in
         let* () = reject_absent "layout" in
         let* () =
           match D.find_arg node "device" with
           | None | Some (Argument.None _) -> return ()
           | Some (Argument.Device { Device.type_ = "cpu"; index = None }) ->
               return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    "_to_copy: device must be omitted/None or CPU without an \
                     index")
         in
         let* () =
           match D.find_arg node "pin_memory" with
           | None | Some (Argument.None _) | Some (Argument.Bool false) ->
               return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    "_to_copy: pin_memory must be omitted/None or false")
         in
         let* () = reject_absent "memory_format" in
         build_g ~name:"to_copy" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = to_copy target x_id in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* `type_as(Tensor self, Tensor other) -> Tensor` (method): `self.to(
     other.dtype)`, so this legalizes onto the SAME [Pointwise.To_copy] node
     [_to_copy.default] above builds -- the target comes from [other]'s own
     RUNTIME dtype rather than an explicit scalar argument (there is none),
     restricted to the same three-way bool/float/long domain
     [Pointwise.To_copy] already covers. The corpus's own occurrence is
     float-to-float (an identity, matching [To_copy.Float]'s own compute),
     but the translation is the fully general one regardless. *)
  | "torch.ops.aten.type_as.default" ->
      Some
        (let* self_t = tensor_arg aten_env node "self" in
         let* other_t = tensor_arg aten_env node "other" in
         let* target =
           match Aten_tensor.scalar_type other_t with
           | Aten_scalar_type.Bool -> return Pointwise.To_copy.Bool
           | Aten_scalar_type.Float -> return Pointwise.To_copy.Float
           | Aten_scalar_type.Long -> return Pointwise.To_copy.Long
           | dtype ->
               fail
                 (`Validation_failure
                    (Printf.sprintf
                       "type_as: other's dtype %s is not supported (only \
                        Bool/Float/Long)"
                       (Aten_scalar_type.to_string dtype)))
         in
         let* x = native_of_aten "self" self_t in
         build_g ~name:"type_as" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = to_copy target x_id in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* `copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor`.
     [self]'s VALUE never matters -- real ATen's [copy_]/[copy] fully
     OVERWRITES [self] with [src] broadcast to [self]'s shape, so the result
     depends only on [src]'s values and [self]'s declared shape/dtype, never
     on what [self] held before. [non_blocking] is a device-transfer hint,
     irrelevant to this single-device engine, read-and-discarded the same
     way [implicit] is in [expand.default]'s arm below. Binds directly to
     the EXISTING [Pointwise.Expand] node -- the corpus's own occurrences
     are exclusively the functionalized form of `tensor[idx] = value`
     (`self` is a [select.int] view, immediately followed by a
     [select_scatter.default] writing the result back), where [self] and
     [src] always share one shape (verified: zero broadcasts, zero dtype
     casts across every occurrence in the 100-model corpus), but the
     translation is the fully general one regardless: `Expand{size=self's
     shape}` applied to [src] computes exactly ATen's own broadcast, not an
     overfit to the observed equal-shape case. The same "map onto an
     existing op after translating parameters" route [sub.Tensor]'s scalar
     form takes to [Add_scalar] -- one node, [copy.default]'s own identity
     not preserved in the graph, same as that arm's. *)
  | "torch.ops.aten.copy.default" ->
      Some
        (let* self_t = tensor_arg aten_env node "self" in
         let* (_ : bool) = bool_arg node "non_blocking" in
         let* target =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.of_aten (aten_dims self_t))
         in
         let* src = native_tensor_arg aten_env node "src" in
         build_g ~name:"copy" [ src ] (function
           | [ src_id ] ->
               let open Graph_builder in
               let+ y = expand { Pointwise.Expand.size = target } src_id in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  | "torch.ops.aten.alias.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* x = native_of_aten "self" aten_x in
         build_g ~name:"alias" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = clone x_id in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* [detach(Tensor(a) self) -> Tensor(a)] drops the autograd history and
     nothing else, so under inference it is the same functional identity as
     [alias.default] and binds to the same [Clone] node. *)
  | "torch.ops.aten.detach.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* x = native_of_aten "self" aten_x in
         build_g ~name:"detach" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = clone x_id in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* `expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) ->
     Tensor(a)`. [implicit] is read-and-discarded: ATen's own
     [at::native::expand] (TensorShape.cpp) takes it as a literally UNUSED
     parameter, so decoding it proves the schema shape without pretending it
     affects the result -- the same treatment [cudnn_enabled] gets in the
     batch_norm arm. Binds to a genuinely new op, [Pointwise.Expand]:
     broadcasts [self] to the resolved [size] by reading through
     [Pointwise_binary.broadcast_coord], the same one-sided reduction a
     binary op's own operand already uses. *)
  | "torch.ops.aten._assert_tensor_metadata.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "a" in
         build_g ~name:"assert_tensor_metadata" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ () = discard x_id in
               []
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* `index.Tensor(Tensor self, Tensor?[] indices) -> Tensor`, restricted to
     the evidenced shape family (`.ai/index_tensor_design.md`): [indices] has
     at most [self]'s ATen rank many entries (real ATen implicitly full-slices
     any missing trailing dims), exactly one live (non-None) entry, of ATen
     rank at least 1, at every other listed position [None]. Unlike
     [Native_interp] (metadata-only import, below), this bridge is reached
     with [indices]' live entry ALREADY RESOLVED to a real ATen tensor VALUE
     by [aten_env] -- whatever node produced it, `clone.default` included,
     since ATen's own [clone] preserves both value and dtype exactly. So
     there is no "trace past Clone" step to take here: [Op_bridge.dispatch]
     sees only this one node (no surrounding graph to trace through, unlike
     [Native_interp]'s importer, which retains the whole node list), and
     reading the live entry's real, already-correct value via
     [native_of_aten] is enough on its own -- the round-6 problem that
     trace-back exists to solve is specifically about [Native_interp]'s
     metadata-only import stamping a wrong dtype on a *newly built* [Clone]
     node's OWN output signature, which never happens here. *)
  (* `embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False,
     sparse=False)`: its own node (see [Embedding]). The indices' real ATen rank
     is read from the value, and a non-Long indices tensor is refused here as
     ATen's own lookup requires an integer one. *)
  | _ -> None
