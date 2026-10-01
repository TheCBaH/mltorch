open Graph_ir
module S = Storage_script
module P = C_payload_layout
module W = C_workspace_plan
module I = Wasm.Instr
module R = Loop_wasm_runtime
module L = Loop_wasm_link

let ( @ ) a b = List.rev_append (List.rev a) b
let concat l = List.concat_map Fun.id l

module Placement = struct
  type t = {
    weights : int;
    inputs : int;
    workspace : int;
    outputs : int;
    total : int;
  }
end

type stats = { invocations : int; distinct_kernels : int; module_bytes : int }

type t = {
  module_ : Wasm.Module.t;
  bytes : string;
  weights : P.t;
  inputs : P.t;
  outputs : P.t;
  identity : string;
  workspace : W.t;
  placement : Placement.t;
  stats : stats;
}

type error =
  [ P.error
  | W.error
  | Loop_wasm.error
  | Wasm_check.error
  | `Memory_over_policy of int64
  | `Missing_signature of Tensor_id.t
  | `Output_outside_workspace of Tensor_id.t ]

let pp_error ppf : [< error ] -> unit = function
  | #P.error as e -> P.pp_error ppf e
  | #W.error as e -> W.pp_error ppf e
  | #Loop_wasm.error as e -> Loop_wasm.pp_error ppf e
  | #Wasm_check.error as e -> Wasm_check.pp_error ppf e
  | `Memory_over_policy n ->
      Format.fprintf ppf "%Ld bytes exceed the 2 GiB memory policy" n
  | `Missing_signature id ->
      Format.fprintf ppf "t%d: no tensor signature" (Tensor_id.to_int id)
  | `Output_outside_workspace id ->
      Format.fprintf ppf "t%d: a kernel output outside the workspace"
        (Tensor_id.to_int id)

let default_config : S.Config.t =
  {
    layout = S.Layout.Separate;
    constants = S.Ownership.Borrowed;
    inputs = S.Ownership.Borrowed;
  }

(* Every address and length below is an [i32] immediate, so the whole memory
   must stay inside a positive [int32]. *)
let memory_cap = 0x8000_0000L

let imm n =
  if Int64.compare n 0L >= 0 && Int64.compare n memory_cap < 0 then
    Err.return (I.I32_const (Int64.to_int32 n))
  else Err.fail (`Memory_over_policy n)

module Func_tbl = Hashtbl.Make (struct
  type t = Wasm.Func.t

  let equal = ( = )
  let hash f = Hashtbl.hash_param 200 400 f
end)

(* The kernels, interned by their complete body: two invocations of the same op
   shape share one function whatever their storage offsets, since offsets are
   arguments. *)
module Kernels = struct
  type t = {
    table : int Func_tbl.t;
    mutable order : Loop_wasm.kernel list;
    mutable next : int;
  }

  let create () = { table = Func_tbl.create 64; order = []; next = 0 }

  let intern t (k : Loop_wasm.kernel) =
    match Func_tbl.find_opt t.table k.Loop_wasm.func with
    | Some i -> i
    | None ->
        let i = t.next in
        t.next <- t.next + 1;
        Func_tbl.add t.table k.Loop_wasm.func i;
        t.order <- k :: t.order;
        i

  let all t = List.rev t.order
end

let signature (g : graph) id =
  match Tensor_id.Map.find_opt id g.Graph.tensors with
  | Some sg -> Err.return sg
  | None -> Err.fail (`Missing_signature id)

(* The four [model_run] parameters: weights 0, inputs 1, workspace 2, outputs 3. *)
let outputs_param = 3

let at ~param off =
  let open Err.Syntax in
  let+ off = imm off in
  [ I.Local_get param; off; I.Numeric Wasm_op.I32_add ]

let location = function
  | W.Weights o -> (0, o)
  | W.Inputs o -> (1, o)
  | W.Workspace o -> (2, o)

let zero_fill ptr bytes =
  let open Err.Syntax in
  let+ n = imm bytes in
  ptr @ [ I.I32_const 0l; n; I.Memory_fill ]

(* One invocation: initialise what it owns, call, and record which one failed.
   Kernel [k] is the [k]th function given to the linker. *)
let invocation ws ~position ~kernel (inv : Loop_bundle.invocation)
    (sc : W.Scratch.t) =
  let open Err.Syntax in
  let scratch_at = W.scratch_offset ws in
  let buffers = inv.Loop_bundle.program.Loop_program.buffers in
  let* local = at ~param:2 scratch_at in
  let* args, inits =
    Err.List.fold_left
      (fun (args, inits) (i, ((buf : Loop_buffer.t), edge)) ->
        match
          List.find_opt (fun c -> c.W.Scratch.position = i) sc.W.Scratch.carves
        with
        | Some c ->
            let* p = at ~param:2 (Int64.add scratch_at c.W.Scratch.offset) in
            let* init =
              match c.W.Scratch.fill with
              | W.Scratch.Zero -> zero_fill p c.W.Scratch.bytes
              | W.Scratch.Value v ->
                  let+ count = imm (Int64.div c.W.Scratch.bytes 4L) in
                  p
                  @ [
                      count;
                      I.F64_const (Int64.bits_of_float v);
                      R.call R.Callee.Fill_f32;
                    ]
            in
            Err.return (p :: args, init :: inits)
        | None -> (
            let* loc = W.locate ws edge in
            let param, off = location loc in
            let* p = at ~param off in
            match buf.Loop_buffer.role with
            | Loop_buffer.Input -> Err.return (p :: args, inits)
            | Loop_buffer.Output -> (
                match loc with
                | W.Workspace _ ->
                    let cell =
                      Int64.of_int
                        (Payload.packed_cell_bytes
                           buf.Loop_buffer.sg.Tensor_sig.fmt)
                    in
                    let n =
                      match
                        Vec6.numel_bounded ~limit:(Int64.shift_left 1L 40)
                          buf.Loop_buffer.sg.Tensor_sig.shape
                      with
                      | Ok n -> n
                      | Error _ -> 0L
                    in
                    let+ init = zero_fill p (Int64.mul n cell) in
                    (p :: args, init :: inits)
                | W.Weights _ | W.Inputs _ ->
                    Err.fail (`Output_outside_workspace edge))
            | Loop_buffer.Scratch -> Err.fail (`Edge_unplaced edge)))
      ([], [])
      (List.mapi
         (fun i x -> (i, x))
         (List.combine buffers inv.Loop_bundle.edges))
  in
  Err.return
    (concat (List.rev inits)
    @ local
    @ concat (List.rev args)
    @ [
        L.kernel_call kernel;
        I.If
          ( None,
            [
              I.I32_const 0l;
              I.I32_const (Int32.of_int position);
              I.Store
                ( Wasm.Store.I32_store,
                  {
                    Wasm.Mem_arg.align = 2;
                    offset = Loop_wasm_failure.invocation_offset;
                  } );
              I.I32_const 1l;
              I.Return;
            ],
            [] );
      ])

let align n a = Int64.mul (Int64.div (Int64.add n (Int64.sub a 1L)) a) a

let build ?(simd = false) (b : Loop_bundle.t) : (t, error) Err.t =
  let open Err.Syntax in
  let g = b.Loop_bundle.graph in
  let sigs ids =
    Err.List.map
      (fun id ->
        let+ sg = signature g id in
        (id, sg))
      ids
  in
  let* const_sigs = sigs b.Loop_bundle.constants in
  let* input_sigs = sigs b.Loop_bundle.inputs in
  let* output_sigs = sigs b.Loop_bundle.outputs in
  let* weights = P.create P.Role.Weights const_sigs in
  let* inputs = P.create P.Role.Inputs input_sigs in
  let* outputs = P.create P.Role.Outputs output_sigs in
  (* The module's own bytes: the error record, then constant tables. *)
  let top = ref (align (Int64.of_int Loop_wasm_failure.record_bytes) 16L) in
  let table_alloc ~bytes =
    let off = align !top 8L in
    top := Int64.add off (Int64.of_int bytes);
    Int64.to_int off
  in
  let kernels = Kernels.create () in
  let* compiled =
    Err.List.map
      (fun (inv : Loop_bundle.invocation) ->
        let* k = Loop_wasm.kernel ~simd ~table_alloc inv.Loop_bundle.program in
        let index = Kernels.intern kernels k in
        let local_doubles = Int64.div k.Loop_wasm.local_bytes 8L in
        let+ sc = W.scratch inv ~local_doubles in
        (inv, index, sc))
      b.Loop_bundle.invocations
  in
  let scratch_bytes =
    List.fold_left
      (fun m (_, _, sc) -> Int64.max m sc.W.Scratch.bytes)
      0L compiled
  in
  let* ws = W.create b ~weights ~inputs ~scratch_bytes in
  let* calls =
    Err.List.map
      (fun (position, (inv, kernel, sc)) ->
        invocation ws ~position ~kernel inv sc)
      (List.mapi (fun i x -> (i, x)) compiled)
  in
  let* copies =
    Err.List.map
      (fun (e : P.Entry.t) ->
        let* loc = W.locate ws e.P.Entry.id in
        let param, off = location loc in
        let* src = at ~param off in
        let* dst = at ~param:outputs_param e.P.Entry.offset in
        let+ n = imm e.P.Entry.bytes in
        dst @ src @ [ n; I.Memory_copy ])
      outputs.P.entries
  in
  let all_kernels = Kernels.all kernels in
  let model_run =
    {
      Wasm.Func.type_ =
        {
          Wasm.Func_type.params = [ Wasm_type.I32; I32; I32; I32 ];
          results = [ Wasm_type.I32 ];
        };
      locals = [];
      body = concat calls @ concat copies @ [ I.I32_const 0l ];
    }
  in
  (* The callees: every kernel's, and [model_run]'s own [Fill_f32]. *)
  let callees =
    R.Callee.Fill_f32
    :: List.concat_map (fun k -> k.Loop_wasm.callees) all_kernels
  in
  let imports, funcs, base =
    L.link ~callees
      (List.map (fun k -> k.Loop_wasm.func) all_kernels @ [ model_run ])
  in
  let static_bytes = align !top 64L in
  let w_at = align static_bytes 64L in
  let i_at = align (Int64.add w_at weights.P.length) 64L in
  let ws_at =
    align (Int64.add i_at inputs.P.length) (Int64.max 64L (W.alignment ws))
  in
  let o_at = align (Int64.add ws_at (W.bytes ws)) 64L in
  let total = Int64.add o_at outputs.P.length in
  let* () =
    if Int64.compare total memory_cap > 0 then
      Err.fail (`Memory_over_policy total)
    else Err.return ()
  in
  let pages = Int64.to_int (Int64.div (Int64.add total 65535L) 65536L) in
  let module_ =
    {
      Wasm.Module.imports;
      funcs;
      globals = [];
      memory = Some { Wasm.Memory.min_pages = max 1 pages; max_pages = None };
      exports =
        [
          { Wasm.Export.name = "memory"; kind = Wasm.Export.Memory };
          {
            Wasm.Export.name = "model_run";
            kind = Wasm.Export.Func (base + List.length all_kernels);
          };
        ];
      data = List.concat_map (fun k -> k.Loop_wasm.data) all_kernels;
      customs = [ { Wasm.Custom.name = "abi"; payload = "loop-wasm/1" } ];
    }
  in
  let module_ =
    {
      module_ with
      Wasm.Module.customs =
        module_.Wasm.Module.customs
        @ [
            {
              Wasm.Custom.name = "manifest";
              payload = L.manifest ~callees module_;
            };
          ];
    }
  in
  let* bytes =
    Err.map_error
      (fun (`Wasm_invalid i) -> `Wasm_invalid i)
      (Wasm_encode.module_ module_)
  in
  Err.return
    {
      module_;
      bytes;
      weights;
      inputs;
      outputs;
      identity = Digest.string bytes;
      workspace = ws;
      placement =
        {
          Placement.weights = Int64.to_int w_at;
          inputs = Int64.to_int i_at;
          workspace = Int64.to_int ws_at;
          outputs = Int64.to_int o_at;
          total = Int64.to_int total;
        };
      stats =
        {
          invocations = List.length compiled;
          distinct_kernels = List.length all_kernels;
          module_bytes = String.length bytes;
        };
    }
