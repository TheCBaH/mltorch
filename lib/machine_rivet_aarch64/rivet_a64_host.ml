(* A model bundle on the CPU with its tensors where the caller keeps them.
   Each invocation is published and loaded once, bound through a table
   ({!Rivet_a64_table}): the image is code and constants, and a call passes the
   addresses of the tensors it reads and writes, which are the context's own
   storage, and of its scratch. Nothing is copied per call, and the loaded code
   is immutable, so it could serve any number of contexts; a context owns every
   byte a kernel writes. *)

open Graph_ir
open Machine_ir
open Machine_interp
module Mm = Machine_model.Mir_model
module Art = Machine_model.Mir_artifact
module Rt = Rivet_a64_route
module I = Rivet_a64_image

let ( let* ) = Result.bind

type kernel = {
  view : Mm.Kernel_view.t;
  slots : Rivet_a64_table.slot list;
  record_slot : int;
  sites : Mir_failure.Site_entry.t array;
  loaded : I.t;
  manifest : Rivet_a64_manifest.t;
}

type t = { model : Mm.t; kernels : kernel list }

let model t = t.model
let manifests t = List.map (fun k -> k.manifest) t.kernels
let invocations t = List.length t.kernels

let slot_index slots region =
  let rec go k = function
    | [] -> None
    | (s : Rivet_a64_table.slot) :: rest ->
        if Mir_id.Region.equal s.Rivet_a64_table.region region then Some k
        else go (k + 1) rest
  in
  go 0 slots

(* A loaded image closes when nothing refers to its kernel any more. *)
let close_with kernel = Gc.finalise (fun k -> I.close k.loaded) kernel

let prepare ?check ?(allocation = Rt.Allocation.Reference)
    ?(runtime = Rivet_a64_runtime.Dependency_free) ~pipeline bundle =
  let refuse k (v : Mm.Kernel_view.t) reason =
    {
      Mm.Refusal.invocation = Int32.of_int k;
      node = v.Mm.Kernel_view.invocation.Loop_ir.Loop_bundle.node;
      reason = Mm.Reason.Route reason;
    }
  in
  match Mm.prepare ~pipeline bundle with
  | Error refusals -> Error refusals
  | Ok model -> (
      let results =
        List.mapi
          (fun k (v : Mm.Kernel_view.t) ->
            let sites = Mm.sites v.Mm.Kernel_view.invocation in
            let built =
              let* { Rt.artifact; record } =
                Rt.publish ~allocation ~sites v.Mm.Kernel_view.program
              in
              let* entry, modules =
                Rt.modules ~binding:Rivet_a64_module.Table ~runtime artifact
              in
              let* () =
                match check with
                | None -> Ok ()
                | Some check -> check ~entry modules
              in
              let* loaded = Rt.load ~entry modules in
              let slots = Rivet_a64_table.of_artifact artifact in
              match slot_index slots record with
              | None ->
                  I.close loaded;
                  Error "the failure record is not a table region"
              | Some record_slot ->
                  let manifest =
                    Rivet_a64_manifest.make ~runtime
                      ~binding:Rivet_a64_module.Table artifact
                  in
                  let kernel =
                    { view = v; slots; record_slot; sites; loaded; manifest }
                  in
                  close_with kernel;
                  Ok kernel
            in
            Result.map_error (refuse k v) built)
          (Mm.kernel_views model)
      in
      match
        List.filter_map (function Error r -> Some r | Ok _ -> None) results
      with
      | _ :: _ as refusals -> Error refusals
      | [] -> Ok { model; kernels = List.map Result.get_ok results })

(* {1 Tensors} *)

(* The address of a tensor's cells and their bytes. *)
let tensor_address (Tensor.Tensor t) = I.address t.Tensor.payload.Payload.data

let tensor_bytes (Tensor.Tensor t) =
  let p = t.Tensor.payload in
  Bigarray.Array1.dim p.Payload.data * Payload.cell_bytes p.Payload.fmt

(* [src]'s cells into [dst], of the same signature. *)
let copy_cells (Tensor.Tensor s) (Tensor.Tensor d) =
  let sp = s.Tensor.payload and dp = d.Tensor.payload in
  match (sp.Payload.fmt, dp.Payload.fmt) with
  | Payload.BF16, Payload.BF16 ->
      Bigarray.Array1.blit sp.Payload.data dp.Payload.data;
      true
  | Payload.Bool, Payload.Bool ->
      Bigarray.Array1.blit sp.Payload.data dp.Payload.data;
      true
  | Payload.F16, Payload.F16 ->
      Bigarray.Array1.blit sp.Payload.data dp.Payload.data;
      true
  | Payload.F32, Payload.F32 ->
      Bigarray.Array1.blit sp.Payload.data dp.Payload.data;
      true
  | Payload.F64, Payload.F64 ->
      Bigarray.Array1.blit sp.Payload.data dp.Payload.data;
      true
  | Payload.I16, Payload.I16 ->
      Bigarray.Array1.blit sp.Payload.data dp.Payload.data;
      true
  | Payload.I32, Payload.I32 ->
      Bigarray.Array1.blit sp.Payload.data dp.Payload.data;
      true
  | Payload.I64, Payload.I64 ->
      Bigarray.Array1.blit sp.Payload.data dp.Payload.data;
      true
  | Payload.I8, Payload.I8 ->
      Bigarray.Array1.blit sp.Payload.data dp.Payload.data;
      true
  | _ -> false

module Context = struct
  type host = t

  (* A kernel's scratch and its table, owned by the context. *)
  type bound = {
    kernel : kernel;
    table :
      (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t;
    scratch :
      (int
      * (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t)
      list;
        (** table slot, storage: the regions no tensor binds *)
  }

  type t = {
    host : host;
    tensors : (Tensor_id.t, Tensor.packed) Hashtbl.t;
    bound : bound list;
  }

  let set_word table k v =
    for b = 0 to 7 do
      table.{(8 * k) + b} <-
        Char.chr
          (Int64.to_int
             (Int64.logand (Int64.shift_right_logical v (8 * b)) 0xFFL))
    done

  let bytes n =
    let b = Bigarray.Array1.create Bigarray.char Bigarray.c_layout (max n 1) in
    Bigarray.Array1.fill b '\000';
    b

  let tensor cx id =
    match (Hashtbl.find_opt cx.tensors id, Mm.tensor_sig cx.host.model id) with
    | Some t, Some _ -> Ok t
    | _ -> Error (Mm.Stop.Missing id)

  (* The context's copy of a tensor, made at its first use. *)
  let storage cx id =
    match Hashtbl.find_opt cx.tensors id with
    | Some t -> Ok t
    | None -> (
        match Mm.tensor_sig cx.host.model id with
        | None -> Error (Mm.Stop.Missing id)
        | Some sg -> (
            match Err.payload (Tensor.create_of_sig sg) with
            | Error (`Quant_missing _) ->
                Error
                  (Mm.Stop.Bind
                     (Fmt.str "%a has no quantization" Tensor_id.pp id))
            | Ok t ->
                Hashtbl.replace cx.tensors id t;
                Ok t))

  let write cx id src =
    let* dst = storage cx id in
    if tensor_bytes src <> tensor_bytes dst then
      Error (Mm.Stop.Bind (Fmt.str "%a changed size" Tensor_id.pp id))
    else if copy_cells src dst then Ok ()
    else Error (Mm.Stop.Bind (Fmt.str "%a changed format" Tensor_id.pp id))

  let write_all cx ids lookup =
    List.fold_left
      (fun acc id ->
        let* () = acc in
        match lookup id with
        | Some t -> write cx id t
        | None -> Error (Mm.Stop.Missing id))
      (Ok ()) ids

  let create host ~constants =
    let cx0 = { host; tensors = Hashtbl.create 64; bound = [] } in
    let b = Mm.bundle host.model in
    let* () = write_all cx0 b.Loop_ir.Loop_bundle.constants constants in
    let* () =
      List.fold_left
        (fun acc (inv : Loop_ir.Loop_bundle.invocation) ->
          List.fold_left
            (fun acc (s : Loop_ir.Loop_bundle.synthetic) ->
              let* () = acc in
              write cx0 s.Loop_ir.Loop_bundle.id
                (Tensor.materialize s.Loop_ir.Loop_bundle.shape (fun _ ->
                     s.Loop_ir.Loop_bundle.value)))
            acc inv.Loop_ir.Loop_bundle.synthetics)
        (Ok ()) b.Loop_ir.Loop_bundle.invocations
    in
    (* every edge a kernel binds has its storage before any table is built *)
    let* () =
      List.fold_left
        (fun acc k ->
          List.fold_left
            (fun acc (_, id) ->
              let* () = acc in
              let* _ = storage cx0 id in
              Ok ())
            acc k.view.Mm.Kernel_view.edges)
        (Ok ()) host.kernels
    in
    let* bound =
      List.fold_left
        (fun acc k ->
          let* bound = acc in
          let table = bytes (Rivet_a64_table.bytes k.slots) in
          let* scratch =
            List.fold_left
              (fun acc (i, (s : Rivet_a64_table.slot)) ->
                let* scratch = acc in
                match
                  List.find_map
                    (fun (r, id) ->
                      if Mir_id.Region.equal r s.Rivet_a64_table.region then
                        Some id
                      else None)
                    k.view.Mm.Kernel_view.edges
                with
                | Some id ->
                    let* t = storage cx0 id in
                    if Int64.of_int (tensor_bytes t) <> s.Rivet_a64_table.size
                    then
                      Error
                        (Mm.Stop.Bind
                           (Fmt.str "%a is %d bytes, its region %Ld"
                              Tensor_id.pp id (tensor_bytes t)
                              s.Rivet_a64_table.size))
                    else (
                      set_word table i (tensor_address t);
                      Ok scratch)
                | None ->
                    let store = bytes (Int64.to_int s.Rivet_a64_table.size) in
                    set_word table i (I.address store);
                    Ok ((i, store) :: scratch))
              (Ok [])
              (List.mapi (fun i s -> (i, s)) k.slots)
          in
          Ok ({ kernel = k; table; scratch } :: bound))
        (Ok []) host.kernels
    in
    Ok { cx0 with bound = List.rev bound }

  let invoke ?(poison = false) k (b : bound) =
    if poison then
      List.iter (fun (_, store) -> Bigarray.Array1.fill store '\xA5') b.scratch;
    let status =
      match Err.payload (I.call ~io:b.table b.kernel.loaded) with
      | Ok v -> Ok (Int64.logand v 0xFFFF_FFFFL)
      | Error e -> Error (Mm.Stop.Bind (Fmt.str "%a" I.Error.pp e))
    in
    let* status = status in
    if Int64.equal status 0L then Ok ()
    else
      (* the record the kernel stored, decoded as the interpreters decode it *)
      let record = List.assoc_opt b.kernel.record_slot b.scratch in
      let node =
        b.kernel.view.Mm.Kernel_view.invocation.Loop_ir.Loop_bundle.node
      in
      let status =
        match record with
        | None ->
            Mir_observation.Status.Defect Mir_observation.Defect.Uninitialized
        | Some store -> (
            let memory = Mir_memory.create () in
            let n = Bigarray.Array1.dim store in
            match
              Mir_memory.alloc memory ~size:(Int64.of_int n) ~align:16L ()
            with
            | None ->
                Mir_observation.Status.Defect
                  Mir_observation.Defect.Invalid_program
            | Some key ->
                Mir_memory.write_string memory key ~offset:0L
                  (String.init n (fun i -> store.{i}));
                Mir_record.status memory key ~sites:b.kernel.sites)
      in
      Error (Mm.Stop.At { invocation = Int32.of_int k; node; status })

  let run_prefix ?poison cx ~inputs ~count =
    let bundle = Mm.bundle cx.host.model in
    let* () = write_all cx bundle.Loop_ir.Loop_bundle.inputs inputs in
    let ran = List.filteri (fun k _ -> k < count) cx.bound in
    let* () =
      List.fold_left
        (fun acc (k, b) ->
          let* () = acc in
          invoke ?poison k b)
        (Ok ())
        (List.mapi (fun k b -> (k, b)) ran)
    in
    Ok
      (List.concat_map
         (fun b ->
           List.filter_map
             (fun (r, edge) ->
               if List.mem r b.kernel.view.Mm.Kernel_view.outputs then Some edge
               else None)
             b.kernel.view.Mm.Kernel_view.edges)
         ran)

  (* A tensor as the caller may keep it: a copy of the context's storage. *)
  let read cx id =
    let* src = tensor cx id in
    match Mm.tensor_sig cx.host.model id with
    | None -> Error (Mm.Stop.Missing id)
    | Some sg -> (
        match Err.payload (Tensor.create_of_sig sg) with
        | Error (`Quant_missing _) ->
            Error
              (Mm.Stop.Bind (Fmt.str "%a has no quantization" Tensor_id.pp id))
        | Ok dst ->
            if copy_cells src dst then Ok dst
            else
              Error (Mm.Stop.Bind (Fmt.str "%a changed format" Tensor_id.pp id))
        )

  let run ?poison cx ~inputs =
    let* _ = run_prefix ?poison cx ~inputs ~count:(List.length cx.bound) in
    List.fold_right
      (fun id acc ->
        let* acc = acc in
        let* t = read cx id in
        Ok (t :: acc))
      (Mm.bundle cx.host.model).Loop_ir.Loop_bundle.outputs (Ok [])

  let tensor = read
end
