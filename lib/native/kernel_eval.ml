(* See kernel_eval.mli. *)

module Binding_mismatch = struct
  type kind =
    | Shape
    | Format
    | Quant
    | Storage_length of { expected : int; actual : int }

  type t = { id : Tensor_id.t; kind : kind }

  let pp fmt { id; kind } =
    match kind with
    | Shape -> Fmt.pf fmt "%a: bound tensor has the wrong shape" Tensor_id.pp id
    | Format ->
        Fmt.pf fmt "%a: bound tensor has the wrong format" Tensor_id.pp id
    | Quant ->
        Fmt.pf fmt "%a: bound tensor has the wrong quantization" Tensor_id.pp id
    | Storage_length { expected; actual } ->
        Fmt.pf fmt "%a: bound tensor holds %d cells, expected %d" Tensor_id.pp
          id actual expected
end

type error =
  [ Expr.Eval.error
  | `Unbound_input of Tensor_id.t
  | `Binding_mismatch of Binding_mismatch.t
  | `Unknown_value of Tensor_id.t ]

let pp_error fmt : [< error ] -> unit = function
  | `Unbound_input id -> Fmt.pf fmt "no binding for input %a" Tensor_id.pp id
  | `Binding_mismatch m -> Binding_mismatch.pp fmt m
  | `Unknown_value id -> Fmt.pf fmt "%a names no value" Tensor_id.pp id
  | #Expr.Eval.error as e -> Expr.Eval.pp_error fmt e

(* ---- the per-run input environment ----------------------------------------

   Resolved and validated BEFORE anything is evaluated, which is what makes
   [`Unbound_input] and [`Binding_mismatch] expressible at all:
   [Expr.Eval.Env.t.load] has the fixed result type
   [(float, Expr.Eval.error) Core.result], and this row is wider. With a
   validated total map, [Expr_bridge.env] can only ever report a genuine
   [`Unknown_source]. *)

let same_shape a b =
  List.for_all
    (fun ax -> Dim.equal (Vec6.get a ax) (Vec6.get b ax))
    Expr.Axis.all

let storage_cells (Tensor.Tensor t) =
  Bigarray.Array1.dim t.Tensor.payload.Payload.data

let payload_quant (Tensor.Tensor t) =
  match t.Tensor.payload.Payload.quant with
  | Payload.No_quant -> None
  | Payload.Quant q -> Some q

(* Shape, format and quantization must match what the kernel DECLARES, and the
   dense payload must actually hold numel cells.

   None of this is bookkeeping. [Expr_bridge.env] reads the BOUND tensor's shape
   and payload, never the declared [Tensor_sig.t], so a wrong-but-present
   binding is silently decoded under a different format or quantization rule and
   the bounds proved for the signature never apply to the tensor actually read.
   The length check is what keeps reads exception-safe: there is no tensor.mli,
   so [Tensor.packed] is a public record and a caller can build one whose shape
   matches while its Bigarray is short — [Tensor.read] bounds-checks the
   COORDINATE against the shape and then indexes [data.{i}], which raises out of
   [Payload.get_float] at an otherwise in-range offset. *)
let check_binding id (sg : Tensor_sig.t) tensor =
  let mismatch kind =
    Core.fail (`Binding_mismatch { Binding_mismatch.id; kind })
  in
  let (Tensor.Tensor t) = tensor in
  if not (same_shape sg.Tensor_sig.shape t.Tensor.shape) then
    mismatch Binding_mismatch.Shape
  else
    let fmt_name (Payload.Fmt f) = Payload.fmt_name f in
    if
      not
        (String.equal
           (fmt_name sg.Tensor_sig.fmt)
           (Payload.fmt_name t.Tensor.payload.Payload.fmt))
    then mismatch Binding_mismatch.Format
    else
      let quant_ok =
        match (sg.Tensor_sig.quant, payload_quant tensor) with
        | None, None -> true
        | Some a, Some b -> Quant.equal a b
        | Some _, None | None, Some _ -> false
      in
      if not quant_ok then mismatch Binding_mismatch.Quant
      else
        let expected = (Vec6.numel sg.Tensor_sig.shape :> int) in
        let actual = storage_cells tensor in
        if expected <> actual then
          mismatch (Binding_mismatch.Storage_length { expected; actual })
        else Core.return ()

let input_env (k : Kernel.t) ~bind =
  let open Core.Syntax in
  List.fold_left
    (fun acc (i : Kernel.Input.t) ->
      let* m = acc in
      match i.Kernel.Input.binding with
      | Kernel.Binding.Filled v ->
          (* Materialised here, exactly as [Stage_program.ground] does, NOT
             handed through as an OCaml float: the store is f32, so a fill that
             is not representable there must be rounded before any consumer
             reads it. *)
          Core.return
            (Tensor_id.Map.add i.Kernel.Input.id
               (Tensor.materialize i.Kernel.Input.sg.Tensor_sig.shape (fun _ ->
                    v))
               m)
      | Kernel.Binding.Caller | Kernel.Binding.Captured_constant -> (
          match bind i.Kernel.Input.id with
          | None -> Core.fail (`Unbound_input i.Kernel.Input.id)
          | Some tensor ->
              let+ () =
                check_binding i.Kernel.Input.id i.Kernel.Input.sg tensor
              in
              Tensor_id.Map.add i.Kernel.Input.id tensor m))
    (Core.return Tensor_id.Map.empty)
    k.Kernel.inputs

(* ---- evaluation ------------------------------------------------------------

   [Tensor.materialize] takes [Vec6.coord -> float] and cannot carry a result,
   so the callback signals through a private exception carrying the ERROR VALUE
   and the boundary converts once. Reusing [Schedule.ground] instead would put
   [Core.or_raise] on the path and let [Expr.Eval] failures escape as exceptions
   through an API promising [Core.result]. The original error is re-returned,
   never rebuilt by hand, so its detection backtrace survives. *)

exception Failed of error Core.Error.t

let caught f = try f () with Failed e -> Error e
let or_raise = function Ok v -> v | Error e -> raise (Failed e)
let widen r = Core.map_error (fun (e : Expr.Eval.error) -> (e :> error)) r

let coord_key (c : int Expr.Coord.t) =
  ( c.Expr.Coord.n,
    c.Expr.Coord.t,
    c.Expr.Coord.d,
    c.Expr.Coord.h,
    c.Expr.Coord.w,
    c.Expr.Coord.c )

let values_by_id (k : Kernel.t) =
  List.fold_left
    (fun m (v : Kernel.Value.t) -> Tensor_id.Map.add v.Kernel.Value.id v m)
    Tensor_id.Map.empty k.Kernel.values

(* The value as its consumers see it: the body wrapped in its result conversion,
   applied exactly once. Built once per value rather than per coordinate. *)
let converted (v : Kernel.Value.t) =
  Kernel.Result_conversion.apply v.Kernel.Value.result v.Kernel.Value.body

let in_shape (sg : Tensor_sig.t) (c : int Expr.Coord.t) =
  List.find_opt
    (fun a ->
      let i = Expr.Coord.get c a in
      i < 0 || i >= Dim.to_int (Vec6.get sg.Tensor_sig.shape a))
    Expr.Axis.all

let value_at k ~bind id coord =
  caught @@ fun () ->
  let inputs = or_raise (input_env k ~bind) in
  let values = values_by_id k in
  let bodies = Tensor_id.Map.map converted values in
  let memo = Hashtbl.create 64 in
  (* Recursion, not buffers: this is the on-demand path, and the one a virtual
     placement uses. [Expr.Eval] is pure, so memoising cannot reorder it. *)
  let rec at id coord =
    match Tensor_id.Map.find_opt id values with
    | None -> raise (Failed (Core.Error.make (`Unknown_value id)))
    | Some (v : Kernel.Value.t) -> (
        match in_shape v.Kernel.Value.sg coord with
        | Some a ->
            raise
              (Failed
                 (Core.Error.make
                    (`Coord_out_of_range
                       ( Expr_bridge.source_of_id id,
                         a,
                         Expr.Coord.get coord a,
                         coord ))))
        | None -> (
            let key = (Tensor_id.to_int id, coord_key coord) in
            match Hashtbl.find_opt memo key with
            | Some v -> v
            | None ->
                let r =
                  or_raise
                    (widen
                       (Expr.Eval.value (env ()) ~output:coord
                          (Tensor_id.Map.find id bodies)))
                in
                Hashtbl.add memo key r;
                r))
  and env () =
    {
      Expr.Eval.Env.load =
        (fun src c ->
          let id = Expr_bridge.id_of_source src in
          if Tensor_id.Map.mem id values then Core.return (at id c)
          else
            (Expr_bridge.env ~binding:(fun i -> Tensor_id.Map.find_opt i inputs))
              .Expr.Eval.Env.load src c);
    }
  in
  Ok (at id coord)

let run k ~bind =
  caught @@ fun () ->
  let inputs = or_raise (input_env k ~bind) in
  let bound = ref inputs in
  let results =
    List.fold_left
      (fun results (v : Kernel.Value.t) ->
        let env =
          (* Every load — boundary and internal alike — goes through
             [Expr_bridge.env] against the same map, so an internal one READS
             the producer's buffer. Materialised values join the map as they are
             computed, which is what makes this the buffer-based path. *)
          Expr_bridge.env ~binding:(fun i -> Tensor_id.Map.find_opt i !bound)
        in
        let body = converted v in
        let t =
          Tensor.materialize v.Kernel.Value.sg.Tensor_sig.shape (fun c ->
              or_raise
                (widen
                   (Expr.Eval.value env
                      ~output:
                        (Expr_bridge.coord_of_vec6 (Vec6.map Dim.to_int c))
                      body)))
        in
        bound := Tensor_id.Map.add v.Kernel.Value.id t !bound;
        Tensor_id.Map.add v.Kernel.Value.id t results)
      Tensor_id.Map.empty k.Kernel.values
  in
  Ok results
