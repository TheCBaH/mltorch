(* See symbolic2.mli. *)

module Make () = struct
  type t = Expr.Value.t
  type 'role index = 'role Expr.Index.t
  type input = Tensor_sig.t
  type b = Expr.Bool.t

  let const = Expr.Value.const
  let add = Expr.Value.add
  let sub = Expr.Value.sub
  let mul = Expr.Value.mul
  let div = Expr.Value.div
  let exp = Expr.Value.exp
  let sqrt = Expr.Value.sqrt
  let lt = Expr.Bool.value_lt
  let select = Expr.Value.select
  let index_zero = Expr.Index.zero
  let index_extent (e : Dim.extent Dim.t) = Expr.Index.const (e :> int)
  let index_const = Expr.Index.const
  let of_index = Expr.Index.of_position
  let index_add = Expr.Index.add
  let index_scale = Expr.Index.scale
  let index_min = Expr.Index.min
  let index_eq = Expr.Bool.index_eq
  let clamp_low = Expr.Index.clamp_low
  let assume_index = Expr.Index.assume_position
  let value_of_index = Expr.Value.value_of_index

  (* [SEMANTICS] returns a plain index while the smart constructor is
     result-returning, so the wrapper is dropped here -- deliberately, and
     through the repository's NAMED boundary rather than an open-coded match, so
     it stays distinguishable from the defect that silently discards a
     backtrace. It is unreachable in practice: [Op_config.Pos.t] is already
     validated positive, which is the whole point of the type. *)
  let to_index r = Core.or_raise Expr.Index.pp_error r

  let index_floor_div_pos a (d : Op_config.Pos.t) =
    to_index (Expr.Index.floor_div_pos a (d :> int))

  let index_ceil_div_pos a (d : Op_config.Pos.t) =
    to_index (Expr.Index.ceil_div_pos a (d :> int))

  let load (s : input) v =
    Expr.Value.load
      (Expr_bridge.source_of_id s.Tensor_sig.id)
      (Expr_bridge.coord_of_vec6 v)

  let load6 (s : input) ~n ~t ~d ~h ~w ~c =
    Expr.Value.load
      (Expr_bridge.source_of_id s.Tensor_sig.id)
      (Expr.Coord.make ~n ~t ~d ~h ~w ~c)

  (* The descriptor carries [in_h]/[in_w] explicitly, because an opaque source
     cannot be asked for its shape the way the embedded [Tensor_sig.t] could.
     They come from [~x_shape], which the older instance ignores precisely
     because it stashed the signature instead. *)
  let max_pool (input : input) ~(x_shape : Vec6.shape) ~kernel ~stride ~pad out
      result =
    (* Extracted per field rather than through a shared helper: the three [Hw.t]s
       carry different element types ([Dim.extent Dim.t], [Pos.t], [Nonneg.t]),
       so one helper would be monomorphised at whichever came first. *)
    let kernel_h = (kernel.Op_config.Hw.h : Dim.extent Dim.t :> int)
    and kernel_w = (kernel.Op_config.Hw.w : Dim.extent Dim.t :> int)
    and stride_h = (stride.Op_config.Hw.h : Op_config.Pos.t :> int)
    and stride_w = (stride.Op_config.Hw.w : Op_config.Pos.t :> int)
    and pad_h = (pad.Op_config.Hw.h : Op_config.Nonneg.t :> int)
    and pad_w = (pad.Op_config.Hw.w : Op_config.Nonneg.t :> int) in
    Expr.Value.intrinsic
      (Core.or_raise Expr.Intrinsic.pp_error
         (Expr.Intrinsic.max_pool
            ~source:(Expr_bridge.source_of_id input.Tensor_sig.id)
            ~in_h:(Dim.to_int (Vec6.get x_shape Axis.H))
            ~in_w:(Dim.to_int (Vec6.get x_shape Axis.W))
            ~kernel_h ~kernel_w ~stride_h ~stride_w ~pad_h ~pad_w
            ~out:(Expr_bridge.coord_of_vec6 out)
            ~result))

  let max_pool2d input ~x_shape ~kernel ~stride ~pad out =
    max_pool input ~x_shape ~kernel ~stride ~pad out
      Expr.Intrinsic.Max_pool.Value

  let max_pool2d_index input ~x_shape ~kernel ~stride ~pad out =
    max_pool input ~x_shape ~kernel ~stride ~pad out
      Expr.Intrinsic.Max_pool.Index

  (* [SEMANTICS.sum] is a plain function, so this instance cannot be written
     monadically. It holds the supply in a ref and advances it through
     [run_from] -- encapsulated mutation over an opaque state, which the design
     permits for exactly this adapter case. It still cannot mint a
     [Reduce_var.t] itself; only [Builder] can. Without [run_from] every
     reduction would restart at ordinal 0 and nested ones would collide. *)
  let supply = ref Expr.Builder.initial

  (* Mint and PUBLISH the advanced supply before evaluating the body. The body
     may itself contain a nested reduction, which reads this same ref -- doing it
     the other way round hands both the same ordinal and the reducers collapse
     onto one variable, which is a wrong answer and not merely a naming clash.
     [Builder.reduction_of] exists to keep that ordering explicit. *)
  let reduce ~kind ~lo ~hi f =
    let var, s = Expr.Builder.run_from !supply Expr.Builder.fresh_reduce in
    supply := s;
    let body = f (Expr.Index.reduce var) in
    Expr.Builder.reduction_of ~kind ~var ~lo ~hi ~body

  let sum ~lo ~hi f = reduce ~kind:Expr.Reduction.Sum ~lo ~hi f
  let max_reduce ~lo ~hi f = reduce ~kind:Expr.Reduction.Max ~lo ~hi f
end

let out_vec : Semantics.position Expr.Index.t Vec6.t =
  Vec6.make ~n:(Expr.Index.output Axis.N) ~t:(Expr.Index.output Axis.T)
    ~d:(Expr.Index.output Axis.D) ~h:(Expr.Index.output Axis.H)
    ~w:(Expr.Index.output Axis.W) ~c:(Expr.Index.output Axis.C)
