open Js_of_ocaml
open Graph_ir
open Loop_ir
module S = Storage_script

type error =
  [ Loop_bundle_js.error
  | Loop_js_exec.error
  | `No_free_arena
  | `Released
  | `Unknown_output of Tensor_id.t
  | `Invocation_failed of Node_id.t * Loop_js_exec.error
  | `Unsupported_forwarded_output of Tensor_id.t ]

let pp_error ppf : [< error ] -> unit = function
  | `No_free_arena ->
      Fmt.pf ppf "every execution arena is running or holds a live result"
  | `Released -> Fmt.pf ppf "the result lease was already released"
  | `Unknown_output id ->
      Fmt.pf ppf "t%d is not an output of this bundle" (Tensor_id.to_int id)
  | `Invocation_failed (n, e) ->
      Fmt.pf ppf "node %a: %a" Node_id.pp n Loop_js_exec.pp_error e
  | `Unsupported_forwarded_output id ->
      Fmt.pf ppf "t%d: a graph output no invocation produces"
        (Tensor_id.to_int id)
  | #Loop_bundle_js.error as e -> Loop_bundle_js.pp_error ppf e
  | #Loop_js_exec.error as e -> Loop_js_exec.pp_error ppf e

type loc =
  | Pooled of { pool : int; off : int; numel : int }
      (** a slice of [slot.pools.(pool)] *)
  | External of int
      (** a borrowed tensor: index into the entry's external parameters *)

type edge = { buffer : Loop_buffer.t; loc : loc }
type slot_state = Free | Running | Leased

(* One execution arena set: every pool but the constants', which all slots
   share. A slot is pinned from a run's start until its lease is released. *)
type slot = { pools : Js.Unsafe.any array; mutable state : slot_state }

type stats = {
  invocations : int;
  distinct_kernels : int;
  source_bytes : int;
  pools : (S.Arena_id.t * Alloc_script.Kind.t * int) list;
  execution_sets : int;
}

type prepared = {
  stats : stats;
  entry : Js.Unsafe.any;
  slots : slot array;
  programs : (Node_id.t * Loop_program.t) array;
  inputs : edge list;
  outputs : edge list;
  externals : edge array;  (** borrowed inputs and constants, entry order *)
  borrowed_constants : Tensor.packed Tensor_id.Map.t;
}

type lease = {
  owner : prepared;
  slot : slot;
  views : Js.Unsafe.any array;  (** this run's borrowed views *)
  mutable live : bool;
}

let global = Js.Unsafe.global
let get = Js.Unsafe.get
let inject = Js.Unsafe.inject
let num n = inject (float_of_int n)

let index_where f l =
  let rec go i = function
    | [] -> None
    | x :: _ when f x -> Some i
    | _ :: rest -> go (i + 1) rest
  in
  go 0 l

let index_of pools arena kind =
  let rec go i = function
    | [] -> None
    | (a, k) :: _
      when S.Arena_id.equal a arena && Alloc_script.Kind.equal k kind ->
        Some i
    | _ :: rest -> go (i + 1) rest
  in
  go 0 pools

(* The first buffer each edge is declared with, by role: a graph input is read
   by some kernel, a graph output written by exactly one. *)
let buffer_of (b : Loop_bundle.t) role id =
  List.find_map
    (fun (inv : Loop_bundle.invocation) ->
      List.find_map
        (fun ((buf : Loop_buffer.t), edge) ->
          if Tensor_id.equal edge id && buf.Loop_buffer.role = role then
            (* The graph's id, not the program-local one, which a Region
               program mints past its sources' and so can collide. *)
            Some { buf with Loop_buffer.id }
          else None)
        (List.combine inv.Loop_bundle.program.Loop_program.buffers
           inv.Loop_bundle.edges))
    b.Loop_bundle.invocations

let edge b (js : Loop_bundle_js.t) role id =
  let open Err.Syntax in
  match buffer_of b role id with
  | None -> Err.return None
  | Some buffer -> (
      match index_where (Tensor_id.equal id) js.Loop_bundle_js.externals with
      | Some i -> Err.return (Some { buffer; loc = External i })
      | None -> (
          let* arena, kind, off, numel = Loop_bundle_js.locate b id in
          match index_of js.Loop_bundle_js.pools arena kind with
          | None -> Err.fail (`Unbound_arena id)
          | Some pool ->
              Err.return (Some { buffer; loc = Pooled { pool; off; numel } })))

let check (e : edge) tensor =
  match
    Err.payload
      (Kernel_eval.check_binding e.buffer.Loop_buffer.id e.buffer.Loop_buffer.sg
         tensor)
  with
  | Ok () -> Ok ()
  | Error (`Binding_mismatch m) -> Error (`Binding_mismatch m)

(* A pooled edge is copied into its slot; a borrowed one is left where it is. *)
let copy_in (slot : slot) (e : edge) tensor =
  match e.loc with
  | External _ -> Ok ()
  | Pooled { pool; off; _ } -> (
      match Loop_js_exec.argument e.buffer tensor with
      | Error (`Binding_mismatch _) as e -> e
      | Ok view ->
          ignore
            (Js.Unsafe.meth_call slot.pools.(pool) "set" [| view; num off |]);
          Ok ())

let prepare ?(max_outstanding = 1) (b : Loop_bundle.t) ~constants :
    (prepared, error) Err.t =
  let open Err.Syntax in
  let* js = Loop_bundle_js.build b |> Err.map_error (fun e -> (e :> error)) in
  let* () =
    if Loop_js_exec.little_endian then Err.return ()
    else Err.fail (`Js_exception "a big-endian host is not supported")
  in
  let* sized =
    Err.List.map
      (fun (arena, kind) ->
        let+ n = Loop_bundle_js.pool_numel b arena kind in
        (arena, kind, n))
      js.Loop_bundle_js.pools
  in
  let make_pool (kind, n) =
    Js.Unsafe.new_obj
      (get global (Loop_bundle_js.typed_array_name kind))
      [| num n |]
  in
  let shared =
    List.map
      (fun (arena, kind, n) ->
        if S.Arena_id.equal arena S.Arena_id.Constants then
          Some (make_pool (kind, n))
        else None)
      sized
  in
  let make_slot () =
    {
      pools =
        Array.of_list
          (List.map2
             (fun (_, kind, n) shared ->
               match shared with Some a -> a | None -> make_pool (kind, n))
             sized shared);
      state = Free;
    }
  in
  let slots = Array.init (max 1 max_outstanding) (fun _ -> make_slot ()) in
  let* entry =
    Loop_js_exec.compile_source
      (Js_print.factory_body js.Loop_bundle_js.program)
  in
  let* inputs =
    Err.List.filter_map (edge b js Loop_buffer.Input) b.Loop_bundle.inputs
  in
  let* consts =
    Err.List.filter_map (edge b js Loop_buffer.Input) b.Loop_bundle.constants
  in
  let* outputs =
    Err.List.map
      (fun id ->
        let* e = edge b js Loop_buffer.Output id in
        match e with
        | Some e -> Err.return e
        | None -> (
            (* Forwarded: no invocation writes it, so it is a graph input or
               constant already resident in its slot; copy it out from there,
               declared as its consumer declares it. *)
            let* fwd = edge b js Loop_buffer.Input id in
            match fwd with
            | Some e ->
                Err.return
                  {
                    e with
                    buffer =
                      { e.buffer with Loop_buffer.role = Loop_buffer.Output };
                  }
            | None -> Err.fail (`Unsupported_forwarded_output id)))
      b.Loop_bundle.outputs
  in
  let ext_edges =
    Array.of_list
      (List.filter
         (fun e -> match e.loc with External _ -> true | Pooled _ -> false)
         (inputs @ consts))
  in
  let ext_edges =
    Array.of_list
      (List.map
         (fun id ->
           List.find
             (fun (e : edge) -> Tensor_id.equal e.buffer.Loop_buffer.id id)
             (Array.to_list ext_edges))
         js.Loop_bundle_js.externals)
  in
  let* borrowed_constants =
    Err.List.fold_left
      (fun acc (e : edge) ->
        match e.loc with
        | Pooled _ -> Err.return acc
        | External _ -> (
            let id = e.buffer.Loop_buffer.id in
            match constants id with
            | None -> Err.fail (`Unbound_input id)
            | Some t -> (
                match check e t with
                | Error m -> Err.fail (m :> error)
                | Ok () -> Err.return (Tensor_id.Map.add id t acc))))
      Tensor_id.Map.empty consts
  in
  let stats =
    {
      invocations = List.length b.Loop_bundle.invocations;
      distinct_kernels = js.Loop_bundle_js.distinct_kernels;
      source_bytes =
        String.length (Js_print.factory_body js.Loop_bundle_js.program);
      pools = sized;
      execution_sets = Array.length slots;
    }
  in
  let p =
    {
      stats;
      externals = ext_edges;
      borrowed_constants;
      entry;
      slots;
      programs =
        Array.of_list
          (List.map
             (fun (inv : Loop_bundle.invocation) ->
               (inv.Loop_bundle.node, inv.Loop_bundle.program))
             b.Loop_bundle.invocations);
      inputs;
      outputs;
    }
  in
  let+ () =
    Err.List.iter
      (fun (e : edge) ->
        let id = e.buffer.Loop_buffer.id in
        match (e.loc, constants id) with
        | External _, _ -> Err.return ()
        | Pooled _, None -> Err.fail (`Unbound_input id)
        | Pooled _, Some t -> (
            match check e t with
            | Error m -> Err.fail (m :> error)
            | Ok () -> (
                (* The constants pool is shared, so any slot's view of it is
                   the one copy. *)
                match copy_in slots.(0) e t with
                | Ok () -> Err.return ()
                | Error m -> Err.fail (m :> error))))
      consts
  in
  p

let run_slot p slot ~bind =
  let ( let* ) = Result.bind in
  let* bound =
    List.fold_left
      (fun acc (e : edge) ->
        let* acc = acc in
        let id = e.buffer.Loop_buffer.id in
        match bind id with
        | None -> Error (`Unbound_input id)
        | Some t ->
            let* () = check e t in
            Ok ((e, t) :: acc))
      (Ok []) p.inputs
  in
  let* () =
    List.fold_left
      (fun acc (e, t) ->
        let* () = acc in
        copy_in slot e t)
      (Ok ()) bound
  in
  let tensor_of (e : edge) =
    let id = e.buffer.Loop_buffer.id in
    match Tensor_id.Map.find_opt id p.borrowed_constants with
    | Some t -> Some t
    | None ->
        List.assoc_opt id
          (List.map (fun (e, t) -> (e.buffer.Loop_buffer.id, t)) bound)
  in
  let* views =
    Array.fold_right
      (fun (e : edge) acc ->
        let* acc = acc in
        match tensor_of e with
        | None -> Error (`Unbound_input e.buffer.Loop_buffer.id)
        | Some t ->
            let* v = Loop_js_exec.argument e.buffer t in
            Ok (v :: acc))
      p.externals (Ok [])
  in
  let views = Array.of_list views in
  match Js.Unsafe.fun_call p.entry (Array.append slot.pools views) with
  | exception Js.Js_error.Exn e ->
      Error (`Js_exception (Js.Js_error.to_string e))
  | r when Js.Unsafe.equals r (inject Js.null) -> Ok views
  | r ->
      let position = int_of_float (Js.float_of_number (Js.Unsafe.get r "0")) in
      let node, program = p.programs.(position) in
      Error
        (`Invocation_failed (node, Loop_js_exec.failure program (get r "1")))

let run_leased p ~bind =
  match Array.find_opt (fun s -> s.state = Free) p.slots with
  | None -> Err.fail `No_free_arena
  | Some slot -> (
      slot.state <- Running;
      match
        try run_slot p slot ~bind
        with e ->
          slot.state <- Free;
          raise e
      with
      | Ok views ->
          slot.state <- Leased;
          Err.return { owner = p; slot; views; live = true }
      | Error e ->
          slot.state <- Free;
          Err.fail (e :> error))

let release l =
  if l.live then (
    l.live <- false;
    l.slot.state <- Free)

let output l id =
  if not l.live then Err.fail `Released
  else
    match
      List.find_opt
        (fun (e : edge) -> Tensor_id.equal e.buffer.Loop_buffer.id id)
        l.owner.outputs
    with
    | None -> Err.fail (`Unknown_output id)
    | Some e -> (
        let out = Loop_interp.allocate e.buffer in
        match Loop_js_exec.argument e.buffer out with
        | Error m -> Err.fail (m :> error)
        | Ok view ->
            let source =
              match e.loc with
              | Pooled { pool; off; numel } ->
                  Js.Unsafe.meth_call l.slot.pools.(pool) "subarray"
                    [| num off; num (off + numel) |]
              | External i -> l.views.(i)
            in
            ignore (Js.Unsafe.meth_call view "set" [| source |]);
            Err.return out)

let run p ~bind =
  let open Err.Syntax in
  let* l = run_leased p ~bind in
  Fun.protect
    ~finally:(fun () -> release l)
    (fun () ->
      let+ outs =
        Err.List.map
          (fun (e : edge) ->
            let id = e.buffer.Loop_buffer.id in
            let+ t = output l id in
            (id, t))
          p.outputs
      in
      List.fold_left
        (fun m (id, t) -> Tensor_id.Map.add id t m)
        Tensor_id.Map.empty outs)

let stats p = p.stats
