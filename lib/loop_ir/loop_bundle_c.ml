open Graph_ir
module S = Storage_script
module P = C_payload_layout
module W = C_workspace_plan

type stats = {
  invocations : int;
  distinct_kernels : int;
  source_bytes : int;
  numerics : Loop_numerics.t;
  f32_invocations : int;
  f32_kernels : int;
  fp32_refusals : (Loop_numerics.Refusal.t * int) list;
}

type t = {
  source : string;
  weights : P.t;
  inputs : P.t;
  outputs : P.t;
  identity : string;
  workspace : W.t;
  stats : stats;
}

type error =
  [ P.error
  | W.error
  | Loop_c.error
  | `Kernel_refused of string
  | `Missing_signature of Tensor_id.t
  | `Output_outside_workspace of Tensor_id.t ]

let pp_error ppf : [< error ] -> unit = function
  | #P.error as e -> P.pp_error ppf e
  | #W.error as e -> W.pp_error ppf e
  | #Loop_c.error as e -> Loop_c.pp_error ppf e
  | `Kernel_refused m -> Format.fprintf ppf "a kernel was refused: %s" m
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

let u64 n = Printf.sprintf "UINT64_C(%Ld)" n

let c_bytes s =
  String.concat ", "
    (List.init (String.length s) (fun i -> string_of_int (Char.code s.[i])))

(* The kernels, interned by their complete text: the entry is emitted under a
   placeholder name, so two invocations of the same op shape share one function
   whatever their storage offsets. *)
module Kernels = struct
  type entry = { index : int; kernel : Loop_c.t }

  type t = {
    table : (string, entry) Hashtbl.t;
    mutable order : entry list;
    mutable next : int;
  }

  let create () = { table = Hashtbl.create 64; order = []; next = 0 }
  let placeholder = "KERNEL"

  let intern ~produce t inv =
    let open Err.Syntax in
    let+ k = produce ~name:placeholder inv in
    match Hashtbl.find_opt t.table k.Loop_c.source with
    | Some e -> e
    | None ->
        let e = { index = t.next; kernel = k } in
        t.next <- t.next + 1;
        Hashtbl.add t.table k.Loop_c.source e;
        t.order <- e :: t.order;
        e

  let name e = Printf.sprintf "kernel_%d" e.index

  (* The placeholder is the definition's name: the text begins with it. *)
  let source e =
    let prefix = "static int " ^ placeholder ^ "(" in
    let s = e.kernel.Loop_c.source in
    if not (String.starts_with ~prefix s) then
      invalid_arg "Loop_bundle_c: a kernel does not begin with its definition";
    "static int " ^ name e ^ "("
    ^ String.sub s (String.length prefix)
        (String.length s - String.length prefix)

  let all t = List.rev t.order
end

let signature (g : graph) id =
  match Tensor_id.Map.find_opt id g.Graph.tensors with
  | Some sg -> Err.return sg
  | None -> Err.fail (`Missing_signature id)

let ptr_expr ~const ty loc =
  let base, off =
    match loc with
    | W.Weights o -> ("w", o)
    | W.Inputs o -> ("in", o)
    | W.Workspace o -> ("ws", o)
  in
  let qual = if const then "const " else "" in
  Printf.sprintf "(%s%s *)(%s + %Ld)" qual ty base off

let scratch_ptr ty ws_scratch (c : W.Scratch.carve) =
  Printf.sprintf "(%s *)(ws + %Ld)" ty (Int64.add ws_scratch c.W.Scratch.offset)

(* One invocation: initialise what it owns, call, and record which invocation
   failed. A kernel receives only [const] views of what it merely reads. *)
let invocation_text ws ~position ~kernel (inv : Loop_bundle.invocation)
    (sc : W.Scratch.t) =
  let open Err.Syntax in
  let scratch_at = W.scratch_offset ws in
  let types = kernel.Kernels.kernel.Loop_c.buffer_types in
  let buffers = inv.Loop_bundle.program.Loop_program.buffers in
  let* args, inits =
    Err.List.fold_left
      (fun (args, inits) (i, ((buf : Loop_buffer.t), edge), ty) ->
        match
          List.find_opt (fun c -> c.W.Scratch.position = i) sc.W.Scratch.carves
        with
        | Some c ->
            let p = scratch_ptr ty scratch_at c in
            let init =
              match c.W.Scratch.fill with
              | W.Scratch.Zero ->
                  Printf.sprintf "    memset(%s, 0, %Ld);" p c.W.Scratch.bytes
              | W.Scratch.Value v ->
                  Printf.sprintf
                    "    { float *p = (float *)(%s); for (size_t k = 0; k < \
                     %Ld; k++) p[k] = (float)%s; }"
                    p
                    (Int64.div c.W.Scratch.bytes 4L)
                    (Loop_c.float_lit v)
            in
            Err.return (p :: args, init :: inits)
        | None -> (
            let* loc = W.locate ws edge in
            match buf.Loop_buffer.role with
            | Loop_buffer.Input ->
                Err.return (ptr_expr ~const:true ty loc :: args, inits)
            | Loop_buffer.Output -> (
                match loc with
                | W.Workspace off ->
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
                    Err.return
                      ( ptr_expr ~const:false ty loc :: args,
                        Printf.sprintf "    memset(ws + %Ld, 0, %Ld);" off
                          (Int64.mul n cell)
                        :: inits )
                | W.Weights _ | W.Inputs _ ->
                    Err.fail (`Output_outside_workspace edge))
            | Loop_buffer.Scratch ->
                (* a scratch buffer always has a carve *)
                Err.fail (`Edge_unplaced edge)))
      ([], [])
      (List.mapi
         (fun i (x, ty) -> (i, x, ty))
         (List.combine (List.combine buffers inv.Loop_bundle.edges) types))
  in
  let local = "(double *)(ws + " ^ Int64.to_string scratch_at ^ ")" in
  Err.return
    (String.concat "\n"
       ([ Printf.sprintf "  { /* invocation %d */" position ]
       @ List.rev inits
       @ [
           Printf.sprintf "    if (%s(error, %s%s)) {" (Kernels.name kernel)
             local
             (String.concat "" (List.map (fun a -> ", " ^ a) (List.rev args)));
           Printf.sprintf "      error->invocation = %d;" position;
           "      return 1;";
           "    }";
           "  }";
         ]))

let build ?vector ?(numerics = Loop_numerics.Reference_f64) ?kernel
    (b : Loop_bundle.t) : (t, error) Err.t =
  let produce ~name (inv : Loop_bundle.invocation) =
    match kernel with
    | Some k -> (
        match k ~name inv with
        | Ok k -> Err.return k
        | Error m -> Err.fail (`Kernel_refused m))
    | None ->
        Err.map_error
          (fun (e : Loop_c.error) -> (e :> error))
          (Loop_c.kernel ?vector ~numerics ~name inv.Loop_bundle.program)
  in
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
  let kernels = Kernels.create () in
  let* compiled =
    Err.List.map
      (fun (inv : Loop_bundle.invocation) ->
        let* e = Kernels.intern ~produce kernels inv in
        let+ sc =
          W.scratch inv ~local_doubles:e.Kernels.kernel.Loop_c.local_doubles
        in
        (inv, e, sc))
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
        invocation_text ws ~position ~kernel inv sc)
      (List.mapi (fun i x -> (i, x)) compiled)
  in
  let* copies =
    Err.List.map
      (fun (e : P.Entry.t) ->
        let+ loc = W.locate ws e.P.Entry.id in
        let src =
          match loc with
          | W.Weights o -> Printf.sprintf "w + %Ld" o
          | W.Inputs o -> Printf.sprintf "in + %Ld" o
          | W.Workspace o -> Printf.sprintf "ws + %Ld" o
        in
        Printf.sprintf "  memcpy(out + %Ld, %s, %Ld);" e.P.Entry.offset src
          e.P.Entry.bytes)
      outputs.P.entries
  in
  let helpers =
    Loop_c_runtime.helpers
      (List.sort_uniq compare
         (List.concat_map
            (fun e -> e.Kernels.kernel.Loop_c.helpers)
            (Kernels.all kernels)))
  in
  let run =
    String.concat "\n"
      ([
         "int model_run(const void *weights, const void *inputs, void \
          *workspace,";
         "              void *outputs, struct model_error *error) {";
         "  const unsigned char *w = (const unsigned char *)weights;";
         "  const unsigned char *in = (const unsigned char *)inputs;";
         "  unsigned char *ws = (unsigned char *)workspace;";
         "  unsigned char *out = (unsigned char *)outputs;";
         "  (void)w; (void)in; (void)ws; (void)out;";
       ]
      @ calls @ copies @ [ "  return 0;"; "}"; "" ])
  in
  let is_f32 (e : Kernels.entry) =
    e.Kernels.kernel.Loop_c.precision = Loop_numerics.Precision.F32
  in
  let f32_invocations =
    List.length (List.filter (fun (_, e, _) -> is_f32 e) compiled)
  in
  let f32_kernels = List.length (List.filter is_f32 (Kernels.all kernels)) in
  let fp32_refusals =
    List.fold_left
      (fun acc (_, (e : Kernels.entry), _) ->
        match e.Kernels.kernel.Loop_c.refusal with
        | None -> acc
        | Some r -> (
            match List.assoc_opt r acc with
            | Some n -> (r, n + 1) :: List.remove_assoc r acc
            | None -> (r, 1) :: acc))
      [] compiled
    |> List.sort compare
  in
  let body =
    String.concat "\n"
      ([ Loop_c_runtime.prelude; C_model_abi.declarations; helpers ]
      @ List.map Kernels.source (Kernels.all kernels)
      @ [
          Printf.sprintf "/* %d invocations, %d distinct kernels */"
            (List.length compiled)
            (List.length (Kernels.all kernels));
          (* The numerical plan is part of the artifact: a kernel's precision is
             in its text, and the policy and its per-precision coverage are
             here, under the identity digest. *)
          Printf.sprintf
            "/* %s; %d of %d invocations and %d of %d kernels binary32 */"
            (Loop_numerics.identity numerics)
            f32_invocations (List.length compiled) f32_kernels
            (List.length (Kernels.all kernels));
          run;
        ])
  in
  (* The identity covers the code and every layout in it; the headers below
     carry it, and it does not cover them. *)
  let identity = Digest.string body in
  let header p = P.header p ~identity in
  let tail =
    String.concat "\n"
      [
        Printf.sprintf "const uint64_t model_weights_size = %s;"
          (u64 weights.P.length);
        Printf.sprintf "const uint64_t model_inputs_size = %s;"
          (u64 inputs.P.length);
        Printf.sprintf "const uint64_t model_outputs_size = %s;"
          (u64 outputs.P.length);
        Printf.sprintf "const uint64_t model_workspace_size = %s;"
          (u64 (W.bytes ws));
        Printf.sprintf "const uint64_t model_workspace_alignment = %s;"
          (u64 (W.alignment ws));
        Printf.sprintf "const unsigned char model_weights_header[64] = {%s};"
          (c_bytes (header weights));
        Printf.sprintf "const unsigned char model_inputs_header[64] = {%s};"
          (c_bytes (header inputs));
        Printf.sprintf "const unsigned char model_outputs_header[64] = {%s};"
          (c_bytes (header outputs));
        "";
      ]
  in
  let source = String.concat "\n" [ body; tail ] in
  Err.return
    {
      source;
      weights;
      inputs;
      outputs;
      identity;
      workspace = ws;
      stats =
        {
          invocations = List.length compiled;
          distinct_kernels = List.length (Kernels.all kernels);
          source_bytes = String.length source;
          numerics;
          f32_invocations;
          f32_kernels;
          fp32_refusals;
        };
    }
