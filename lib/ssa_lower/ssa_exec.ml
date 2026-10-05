open Ssa_ir

type error =
  [ Ssa_interp.failure
  | `Binding_mismatch of Kernel_eval.Binding_mismatch.t
  | `Invalid_program of Ssa_verify.diagnostic
  | `Unbound_input of Tensor_id.t ]

let pp_error fmt : [< error ] -> unit = function
  | #Ssa_interp.error as e -> Ssa_interp.pp_error fmt e
  | `Binding_mismatch m -> Kernel_eval.Binding_mismatch.pp fmt m
  | `Unbound_input id -> Fmt.pf fmt "no binding for input %a" Tensor_id.pp id

let shape_of = Ssa_sig.shape

let elements (b : Ssa_buffer.t) =
  match Ssa_buffer.elements b.Ssa_buffer.extents with
  | Some n -> Int64.to_int n
  | None -> invalid_arg "Ssa_exec: a buffer larger than an index"

(* A bound input's cells, in the row-major order [Vec6.iter] visits, which is
   also the order of the tensor's dense storage. *)
let cells_of (b : Ssa_buffer.t) (Tensor.Tensor t as packed) =
  let shape = shape_of b in
  let n = elements b in
  match Ssa_format.cells b.Ssa_buffer.format with
  | Ssa_format.Float_cells ->
      (* a Bool cell of any nonzero byte reads as 1, a decoded float exactly *)
      let a = Array.make n 0. in
      Vec6.iter shape (fun c ->
          a.((Vec6.offset shape c :> int)) <- Tensor.read packed c);
      Ssa_memory.Floats a
  | Ssa_format.Int64_cells ->
      let a = Array.make n 0L in
      Vec6.iter shape (fun c ->
          a.((Vec6.offset shape c :> int)) <-
            (match
               Tensor.read_i64_at6 packed (fun axis ->
                   Dim.to_int (Vec6.get c axis))
             with
            | Ok v -> v
            | Error _ -> invalid_arg "Ssa_exec: an i64 input is not i64"));
      Ssa_memory.Int64s a
  | Ssa_format.Int_cells -> (
      (* the raw storage cell: 16-bit float bits, a quantized integer or an i32 *)
      match t.Tensor.payload.Payload.fmt with
      | Payload.BF16 ->
          Ssa_memory.Ints
            (Array.init n (fun i -> t.Tensor.payload.Payload.data.{i}))
      | Payload.F16 ->
          Ssa_memory.Ints
            (Array.init n (fun i -> t.Tensor.payload.Payload.data.{i}))
      | Payload.I16 ->
          Ssa_memory.Ints
            (Array.init n (fun i -> t.Tensor.payload.Payload.data.{i}))
      | Payload.I8 ->
          Ssa_memory.Ints
            (Array.init n (fun i -> t.Tensor.payload.Payload.data.{i}))
      | Payload.I32 ->
          Ssa_memory.Ints
            (Array.init n (fun i ->
                 Int32.to_int t.Tensor.payload.Payload.data.{i}))
      | Payload.Bool | Payload.F32 | Payload.F64 | Payload.I64 ->
          invalid_arg "Ssa_exec: cells do not match the input's format")

(* An output's cells as the tensor a caller receives. *)
let tensor_of (b : Ssa_buffer.t) cells =
  let shape = shape_of b in
  let at c = (Vec6.offset shape c :> int) in
  match (b.Ssa_buffer.format, cells) with
  | Ssa_format.F32, Ssa_memory.Floats a ->
      Tensor.materialize shape (fun c -> a.(at c))
  | Ssa_format.Bool, Ssa_memory.Floats a ->
      Tensor.materialize_bool shape (fun c -> a.(at c) <> 0.)
  | Ssa_format.I64, Ssa_memory.Int64s a ->
      Tensor.materialize_i64 shape (fun c -> a.(at c))
  | _ -> invalid_arg "Ssa_exec: cells do not match the buffer format"

let run ?counters ?fused (plan : Fusion_plan.t) (p : Ssa_program.t) ~bind =
  Err.Escape.with_escape @@ fun esc ->
  (* The reference validates every bound input first, in input order, used or
     not, so a failure there is reported before any evaluation. *)
  List.iter
    (fun (i : Kernel.Input.t) ->
      match i.Kernel.Input.binding with
      | Kernel.Binding.Caller | Kernel.Binding.Captured_constant -> (
          let sg = i.Kernel.Input.sg in
          match bind sg.Tensor_sig.id with
          | None ->
              Err.Escape.throw esc (`Unbound_input sg.Tensor_sig.id : error)
          | Some t ->
              Err.Escape.or_throw esc
                (Err.map_error
                   (fun (`Binding_mismatch m) -> `Binding_mismatch m)
                   (Kernel_eval.check_binding sg.Tensor_sig.id sg t)))
      | Kernel.Binding.Filled _ | Kernel.Binding.Filled_i64 _ -> ())
    plan.Fusion_plan.kernel.Kernel.inputs;
  let memory =
    List.fold_left
      (fun m (b : Ssa_buffer.t) ->
        let cells =
          match b.Ssa_buffer.role with
          | Ssa_buffer.Input -> (
              match bind (Tensor_id.of_int (b.Ssa_buffer.id :> int)) with
              | Some t -> cells_of b t
              | None -> invalid_arg "Ssa_exec: a declared input is unbound")
          | Ssa_buffer.Output | Ssa_buffer.Scratch -> Ssa_memory.zeroed b
        in
        Ssa_id.Buffer.Map.add b.Ssa_buffer.id cells m)
      Ssa_id.Buffer.Map.empty p.Ssa_program.buffers
  in
  Err.Escape.or_throw esc
    (Err.map_error
       (fun (e : Ssa_interp.error) -> (e :> error))
       (Ssa_interp.run ?counters ?fused p ~memory));
  List.fold_left
    (fun acc (b : Ssa_buffer.t) ->
      match b.Ssa_buffer.role with
      | Ssa_buffer.Output ->
          let cells = Ssa_id.Buffer.Map.find b.Ssa_buffer.id memory in
          Tensor_id.Map.add
            (Tensor_id.of_int (b.Ssa_buffer.id :> int))
            (tensor_of b cells) acc
      | Ssa_buffer.Input | Ssa_buffer.Scratch -> acc)
    Tensor_id.Map.empty p.Ssa_program.buffers
