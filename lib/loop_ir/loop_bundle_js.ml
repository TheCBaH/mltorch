open Graph_ir
module S = Storage_script
module U = Core.Storage_units

type error =
  [ `Kernel_refused of string
  | `Local_too_large of Tensor_id.t
  | `Pool_index_overflow of Tensor_id.t * int64
  | `Storage_units of Core.Storage_units.error
  | `Unbound_arena of Tensor_id.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Kernel_refused m -> Format.fprintf ppf "a kernel was refused: %s" m
  | `Local_too_large id ->
      Format.fprintf ppf
        "t%d: a kernel-local buffer does not fit a 32-bit index"
        (Tensor_id.to_int id)
  | `Pool_index_overflow (id, v) ->
      Format.fprintf ppf "t%d: pool index %Ld does not fit a 32-bit index"
        (Tensor_id.to_int id) v
  | `Storage_units e -> Core.Storage_units.pp_error ppf e
  | `Unbound_arena id ->
      Format.fprintf ppf "t%d: no arena binding (borrowed/quantized/outside)"
        (Tensor_id.to_int id)

(* [lib/loop_ir] is JS-reachable and js_of_ocaml's [int] is 32 bits: an
   element offset/count is bounded below any real model's pool size today,
   but is still checked here rather than assumed, per repo convention. *)
let to_pool_index id (v : int64) =
  if Int64.abs v <= 2147483647L then Err.return (Int64.to_int v)
  else Err.fail (`Pool_index_overflow (id, v))

let ident = Js_ident.v
let num n = Js_ast.Number (float_of_int n)
let call recv meth args = Js_ast.Call (Js_ast.Member (recv, ident meth), args)

let pool_ident arena kind =
  ident
    (String.lowercase_ascii
       (Fmt.str "pool_%a_%a" S.Arena_id.pp arena Alloc_script.Kind.pp kind))

(* Every arena-backed edge's pool kind, element offset and element count in
   that pool. *)
let resolve (plan : Storage_plan.t) arena_id id =
  let open Err.Syntax in
  let* arena_plan =
    Storage_plan.arena plan arena_id
    |> Err.of_option (`Unbound_arena id : [> error ])
  in
  let* slot =
    Arena_plan.slot arena_plan id |> Err.of_option (`Unbound_arena id)
  in
  let kind = slot.Arena_plan.Slot.kind in
  let* elem_off =
    U.Element_offset.of_bytes slot.Arena_plan.Slot.offset
      (Alloc_script.Kind.element_bytes kind)
    |> Err.map_error (fun e -> `Storage_units e)
  in
  let* off = to_pool_index id (U.Element_offset.to_int64 elem_off) in
  let+ numel =
    to_pool_index id (U.Element_count.to_int64 slot.Arena_plan.Slot.numel)
  in
  (kind, off, numel)

let typed_array_name : Alloc_script.Kind.t -> string = function
  | Alloc_script.Kind.Float32 -> "Float32Array"
  | Float64 -> "Float64Array"
  | Int16_signed -> "Int16Array"
  | Int16_unsigned -> "Uint16Array"
  | Int32 -> "Int32Array"
  | Int64 -> "BigInt64Array"
  | Int8_signed -> "Int8Array"
  | Int8_unsigned -> "Uint8Array"

(* [id]'s local const identifier, allocated (and its binding statement
   returned) the first time it is touched -- as a graph input/constant, or as
   an invocation's own output. A shared edge is bound ONCE: a later read
   reuses the identifier, emitting no statement. *)
type binder = {
  bound : (Tensor_id.t, Js_ident.t) Hashtbl.t;
  arena_of : S.Arena_id.t option Tensor_id.Map.t;
  externals : (Tensor_id.t, Js_ident.t) Hashtbl.t;
      (** borrowed graph inputs/constants: entry parameters, never views *)
  plan : Storage_plan.t;
  mutable next : int;
}

let bind st id =
  let open Err.Syntax in
  match Hashtbl.find_opt st.bound id with
  | Some name -> Err.return (name, [])
  | None when Hashtbl.mem st.externals id ->
      Err.return (Hashtbl.find st.externals id, [])
  | None ->
      let* arena_id =
        match Tensor_id.Map.find_opt id st.arena_of with
        | Some (Some a) -> Err.return a
        | Some None | None -> Err.fail (`Unbound_arena id)
      in
      let+ kind, off, numel = resolve st.plan arena_id id in
      let name = ident (Fmt.str "e%d" st.next) in
      st.next <- st.next + 1;
      Hashtbl.add st.bound id name;
      let stmt =
        Js_ast.Stmt.Const
          ( name,
            call
              (Js_ast.Var (pool_ident arena_id kind))
              "subarray"
              [ num off; num (off + numel) ] )
      in
      (name, [ stmt ])

let zero_fill name kind =
  let zero =
    match kind with Alloc_script.Kind.Int64 -> Js_ast.Bigint 0L | _ -> num 0
  in
  Js_ast.Stmt.Expr (call (Js_ast.Var name) "fill" [ zero ])

let edge_of st id =
  match Tensor_id.Map.find_opt id st.arena_of with
  | Some (Some a) -> resolve st.plan a id
  | Some None | None -> Err.fail (`Unbound_arena id)

(* One invocation's binding statements (its edges, first-touch only) plus its
   call-and-check: [const rK = kernel_I(...); if (rK !== null) return [P, rK];],
   P the invocation's schedule position, so a failure is decoded against
   its own program.
   Buffers are bound in [program.buffers] order, matching the positional call
   the kernel expects. *)
let global_of_typed_array = function
  | "BigInt64Array" -> Js_global.Big_int64_array
  | "Float32Array" -> Js_global.Float32_array
  | "Float64Array" -> Js_global.Float64_array
  | "Int16Array" -> Js_global.Int16_array
  | "Int32Array" -> Js_global.Int32_array
  | "Int8Array" -> Js_global.Int8_array
  | "Uint16Array" -> Js_global.Uint16_array
  | "Uint8Array" -> Js_global.Uint8_array
  | name -> invalid_arg ("Loop_bundle_js: no typed array global " ^ name)

(* A binding local to one invocation: a synthetic default or kernel scratch is a
   fresh typed array of the buffer's own format, never a pool view, so it can
   neither alias an arena slot nor outlive the call. *)
let local_array ~name ~id ~(buf : Loop_buffer.t) ~fill =
  let open Err.Syntax in
  let+ numel =
    Vec6.numel_bounded ~limit:2147483648L buf.Loop_buffer.sg.Tensor_sig.shape
    |> Err.map_error (fun _ -> `Local_too_large id)
  in
  let make =
    Js_ast.New
      ( Js_ast.Global (global_of_typed_array (Loop_js.typed_array buf)),
        [ num (Int64.to_int numel) ] )
  in
  Js_ast.Stmt.Const (name, make)
  ::
  (match fill with
  | None -> []
  | Some v ->
      [ Js_ast.Stmt.Expr (call (Js_ast.Var name) "fill" [ Js_ast.Number v ]) ])

let invocation_stmts st ~kernel_name ~position (inv : Loop_bundle.invocation) =
  let open Err.Syntax in
  let synthetic id =
    List.find_opt
      (fun (s : Loop_bundle.synthetic) -> Tensor_id.equal s.Loop_bundle.id id)
      inv.Loop_bundle.synthetics
  in
  let+ args, bind_stmts =
    Err.List.fold_left
      (fun (args, stmts) ((buf : Loop_buffer.t), edge) ->
        match (synthetic edge, buf.Loop_buffer.role) with
        | Some s, _ ->
            let name = ident (Fmt.str "s%d_%d" position (List.length args)) in
            let+ decl =
              local_array ~name ~id:edge ~buf ~fill:(Some s.Loop_bundle.value)
            in
            (args @ [ Js_ast.Var name ], stmts @ decl)
        | None, Loop_buffer.Scratch ->
            let name = ident (Fmt.str "s%d_%d" position (List.length args)) in
            let+ decl = local_array ~name ~id:edge ~buf ~fill:None in
            (args @ [ Js_ast.Var name ], stmts @ decl)
        | None, Loop_buffer.Output ->
            let* name, new_stmts = bind st edge in
            let+ kind, _, _ = edge_of st edge in
            ( args @ [ Js_ast.Var name ],
              stmts @ new_stmts @ [ zero_fill name kind ] )
        | None, Loop_buffer.Input ->
            let+ name, new_stmts = bind st edge in
            (args @ [ Js_ast.Var name ], stmts @ new_stmts))
      ([], [])
      (List.combine inv.Loop_bundle.program.Loop_program.buffers
         inv.Loop_bundle.edges)
  in
  let result = ident (Fmt.str "r%d" position) in
  bind_stmts
  @ [
      Js_ast.Stmt.Const (result, Js_ast.Call (Js_ast.Var kernel_name, args));
      Js_ast.Stmt.If
        ( Js_ast.Binary (Js_ast.Ne_strict, Js_ast.Var result, Js_ast.Null),
          [
            Js_ast.Stmt.Return
              (Some (Js_ast.Array [ num position; Js_ast.Var result ]));
          ],
          [] );
    ]

let arena_of_script (b : Loop_bundle.t) =
  List.fold_left
    (fun m (e : S.Event.t) ->
      match e with
      | S.Event.Alloc { S.Block.alloc; arena; _ } ->
          Tensor_id.Map.add alloc.Alloc_script.Alloc.id arena m
      | S.Event.Boundary _ | S.Event.Free _ | S.Event.Node _ -> m)
    Tensor_id.Map.empty
    (S.events b.Loop_bundle.script)

let locate (b : Loop_bundle.t) id =
  let open Err.Syntax in
  let* arena =
    match Tensor_id.Map.find_opt id (arena_of_script b) with
    | Some (Some a) -> Err.return a
    | Some None | None -> Err.fail (`Unbound_arena id)
  in
  let+ kind, off, numel = resolve b.Loop_bundle.plan arena id in
  (arena, kind, off, numel)

let pool_numel (b : Loop_bundle.t) arena kind =
  let open Err.Syntax in
  let id = Tensor_id.of_int 0 in
  let* arena_plan =
    Storage_plan.arena b.Loop_bundle.plan arena
    |> Err.of_option (`Unbound_arena id : [> error ])
  in
  match
    List.find_opt
      (fun (p : Arena_plan.Pool.t) -> p.Arena_plan.Pool.kind = kind)
      (Arena_plan.pools arena_plan)
  with
  | None -> Err.fail (`Unbound_arena id)
  | Some p ->
      to_pool_index id (U.Element_count.to_int64 p.Arena_plan.Pool.numel)

type t = {
  program : Js_ast.Program.t;
  distinct_kernels : int;
  pools : (S.Arena_id.t * Alloc_script.Kind.t) list;
  externals : Tensor_id.t list;
      (** borrowed graph inputs/constants, the entry parameters after [pools],
          in order *)
  entry_name : Js_ident.t;
}

let entry_name = ident "run_bundle"

(* [Loop_js_runtime]'s helpers are a fixed, name-keyed global table (see
   [Loop_js.prelude]): two kernels needing the same helper always get the
   textually identical declaration, so a bundle-wide prelude is their union,
   deduped by printed text -- never a per-kernel copy that could (in
   principle) drift. *)
let dedup_stmts stmts =
  let seen = Hashtbl.create 16 in
  List.filter
    (fun s ->
      let text = Js_print.stmts [ s ] in
      if Hashtbl.mem seen text then false
      else (
        Hashtbl.add seen text ();
        true))
    stmts

let build ?kernel (b : Loop_bundle.t) : (t, error) Err.t =
  let open Err.Syntax in
  let arena_of = arena_of_script b in
  let pools =
    List.concat_map
      (fun (arena_id, arena_plan) ->
        List.map
          (fun (p : Arena_plan.Pool.t) -> (arena_id, p.Arena_plan.Pool.kind))
          (Arena_plan.pools arena_plan))
      (Storage_plan.arenas b.Loop_bundle.plan)
  in
  let externals =
    List.filter
      (fun id ->
        match Tensor_id.Map.find_opt id arena_of with
        | Some (Some _) -> false
        | Some None | None -> true)
      (b.Loop_bundle.inputs @ b.Loop_bundle.constants)
  in
  let external_tbl = Hashtbl.create 16 in
  List.iter
    (fun id ->
      Hashtbl.add external_tbl id
        (ident (Fmt.str "ext_%d" (Tensor_id.to_int id))))
    externals;
  let st =
    {
      externals = external_tbl;
      bound = Hashtbl.create 256;
      arena_of;
      plan = b.Loop_bundle.plan;
      next = 0;
    }
  in
  (* Interning: distinct invocations by complete canonical printed source,
     keeping each interned kernel's original [Js_ast.Program.t] the first
     time it is seen (plan T3.3, "verify distinct kernels are not incorrectly
     interned and identical kernels share despite differing storage
     offsets"). Renaming the entry (always "loop_kernel", [Loop_js.function_name])
     to a bundle-unique name is enough to avoid collisions: prelude HELPERS are
     already kernel-independent (see [dedup_stmts]'s own doc), so no per-kernel
     lexical scope is needed to isolate them. *)
  let kernel_table : (string, int * Js_ast.Program.t) Hashtbl.t =
    Hashtbl.create 64
  in
  let next_kernel = ref 0 in
  let kernel_order = ref [] in
  let intern (ast : Js_ast.Program.t) =
    let text = Js_print.factory_body ast in
    match Hashtbl.find_opt kernel_table text with
    | Some (idx, _) -> idx
    | None ->
        let idx = !next_kernel in
        incr next_kernel;
        Hashtbl.add kernel_table text (idx, ast);
        kernel_order := (idx, ast) :: !kernel_order;
        idx
  in
  let kernel_name idx = ident (Fmt.str "kernel_%d" idx) in
  let* head_stmts =
    Err.List.fold_left
      (fun stmts id ->
        let+ _name, new_stmts = bind st id in
        stmts @ new_stmts)
      []
      (b.Loop_bundle.inputs @ b.Loop_bundle.constants)
  in
  let+ body_stmts, _ =
    Err.List.fold_left
      (fun (stmts, position) (inv : Loop_bundle.invocation) ->
        let* ast =
          match kernel with
          | None -> Err.return (Loop_js.to_ast inv.Loop_bundle.program)
          | Some k -> (
              match k inv with
              | Ok ast -> Err.return ast
              | Error m -> Err.fail (`Kernel_refused m))
        in
        let idx = intern ast in
        let+ inv_stmts =
          invocation_stmts st ~kernel_name:(kernel_name idx) ~position inv
        in
        (stmts @ inv_stmts, position + 1))
      ([], 0) b.Loop_bundle.invocations
  in
  let distinct = List.rev !kernel_order in
  let kernel_decls =
    List.map
      (fun (idx, (ast : Js_ast.Program.t)) ->
        Js_ast.Stmt.Function
          { ast.Js_ast.Program.entry with Js_ast.Func.name = kernel_name idx })
      distinct
  in
  let helper_prelude =
    dedup_stmts
      (List.concat_map (fun (_, a) -> a.Js_ast.Program.prelude) distinct)
  in
  let entry =
    {
      Js_ast.Func.name = entry_name;
      params =
        List.map (fun (a, k) -> pool_ident a k) pools
        @ List.map (fun id -> Hashtbl.find external_tbl id) externals;
      body = head_stmts @ body_stmts @ [ Js_ast.Stmt.Return (Some Js_ast.Null) ];
    }
  in
  let program =
    { Js_ast.Program.prelude = helper_prelude @ kernel_decls; entry }
  in
  (match Js_check.closed program with
  | Ok () -> ()
  | Error faults ->
      invalid_arg
        (Fmt.str "Loop_bundle_js.build: the program is not closed: %a"
           Fmt.(list ~sep:(any "; ") Js_check.Fault.pp)
           faults));
  {
    program;
    distinct_kernels = List.length distinct;
    pools;
    externals;
    entry_name;
  }
