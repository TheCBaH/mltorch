open Graph_ir

type error = [ `Failed of int * Loop_interp.error | `Missing of Tensor_id.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Failed (i, e) -> Fmt.pf ppf "invocation %d: %a" i Loop_interp.pp_error e
  | `Missing id -> Fmt.pf ppf "no value for %a" Tensor_id.pp id

let synthetic_tensor (s : Loop_bundle.synthetic) =
  Tensor.materialize s.Loop_bundle.shape (fun _ -> s.Loop_bundle.value)

let run ?(precision = Loop_numerics.Precision.F32) (b : Loop_bundle.t)
    ~constants ~inputs =
  let ( let* ) = Result.bind in
  let env = ref Tensor_id.Map.empty in
  let lookup id =
    match Tensor_id.Map.find_opt id !env with
    | Some t -> Some t
    | None -> ( match constants id with Some t -> Some t | None -> inputs id)
  in
  let invoke position (inv : Loop_bundle.invocation) =
    let buffers = inv.Loop_bundle.program.Loop_program.buffers in
    let edge_of (buf : Loop_buffer.t) =
      List.assoc buf.Loop_buffer.id
        (List.map2
           (fun (x : Loop_buffer.t) e -> (x.Loop_buffer.id, e))
           buffers inv.Loop_bundle.edges)
    in
    let bound =
      List.filter_map
        (fun (buf : Loop_buffer.t) ->
          match buf.Loop_buffer.role with
          | Loop_buffer.Input -> (
              let edge = edge_of buf in
              match
                List.find_opt
                  (fun (s : Loop_bundle.synthetic) ->
                    Tensor_id.equal s.Loop_bundle.id edge)
                  inv.Loop_bundle.synthetics
              with
              | Some s -> Some (Ok (buf.Loop_buffer.id, synthetic_tensor s))
              | None -> (
                  match lookup edge with
                  | Some t -> Some (Ok (buf.Loop_buffer.id, t))
                  | None -> Some (Error (`Missing edge))))
          | Loop_buffer.Output | Loop_buffer.Scratch -> None)
        buffers
    in
    let* bound =
      List.fold_left
        (fun acc r ->
          let* acc = acc in
          let* x = r in
          Ok (x :: acc))
        (Ok []) bound
    in
    let bind id = List.assoc_opt id bound in
    let program = Loop_sum.program inv.Loop_bundle.program in
    match Err.payload (Loop_interp.run ~precision program ~bind) with
    | Error e -> Error (`Failed (position, e))
    | Ok outs ->
        List.iter
          (fun (buf : Loop_buffer.t) ->
            if buf.Loop_buffer.role = Loop_buffer.Output then
              match Tensor_id.Map.find_opt buf.Loop_buffer.id outs with
              | Some t -> env := Tensor_id.Map.add (edge_of buf) t !env
              | None -> ())
          buffers;
        Ok ()
  in
  let result =
    let* () =
      List.fold_left
        (fun acc (position, inv) ->
          let* () = acc in
          invoke position inv)
        (Ok ())
        (List.mapi (fun i inv -> (i, inv)) b.Loop_bundle.invocations)
    in
    List.fold_left
      (fun acc id ->
        let* acc = acc in
        match lookup id with
        | Some t -> Ok (t :: acc)
        | None -> Error (`Missing id))
      (Ok []) b.Loop_bundle.outputs
    |> Result.map List.rev
  in
  match result with Ok v -> Err.return v | Error e -> Err.fail e
