open Err.Syntax
module F = Pt2_fixture
module Dtype = Pt2_checkpoint_map.Dtype
module Report = F.Report

let backend = "native-direct"

(* The route that was executed, as the report states it: a different
   accumulation policy is a different backend row, never a quiet variant. *)
let backend_of = function
  | Direct.Binary64 -> backend
  | Direct.Binary32_sequential -> backend ^ "+binary32-sequential-dots"

(* --- Native tensor -> Logical ------------------------------------------- *)

let dtype_of_payload (Tensor.Tensor t) : Dtype.t option =
  match Payload.Fmt t.Tensor.payload.Payload.fmt with
  | Payload.Fmt Payload.F32 -> Some Dtype.F32
  | Payload.Fmt Payload.F64 -> Some Dtype.F64
  | Payload.Fmt Payload.I64 -> Some Dtype.I64
  | Payload.Fmt Payload.I32 -> Some Dtype.I32
  | Payload.Fmt Payload.Bool -> Some Dtype.BOOL
  | Payload.Fmt _ -> None

let to_logical (Tensor.Tensor t as packed) ~rank =
  let shape = t.Tensor.shape in
  let axes = Array.of_list Axis.all in
  let extent a = (Vec6.get shape a :> int) in
  if rank < 0 || rank > 6 then Error (`Native_layout "rank above six")
  else
    let lead = 6 - rank in
    let leading_ok =
      List.for_all (fun i -> extent axes.(i) = 1) (List.init lead Fun.id)
    in
    if not leading_ok then Error (`Native_layout "leading frame axes are not 1")
    else
      match dtype_of_payload packed with
      | None -> Error (`Native_layout "unsupported payload format")
      | Some dtype ->
          let used = List.init rank (fun i -> axes.(lead + i)) in
          let dims = List.map extent used in
          let n = List.fold_left ( * ) 1 dims in
          let width = Dtype.byte_width dtype in
          let out =
            Bigarray.Array1.create Bigarray.char Bigarray.c_layout (n * width)
          in
          let index = Array.make rank 0 in
          let coord () =
            let c = Array.make 6 0 in
            List.iteri
              (fun k a ->
                let pos = ref 0 in
                Array.iteri (fun i b -> if b = a then pos := i) axes;
                c.(!pos) <- index.(k))
              used;
            Vec6.coord ~n:c.(0) ~t:c.(1) ~d:c.(2) ~h:c.(3) ~w:c.(4) ~c:c.(5)
          in
          let put i bytes =
            String.iteri
              (fun b ch -> Bigarray.Array1.unsafe_set out ((i * width) + b) ch)
              bytes
          in
          let le64 v =
            let b = Bytes.create 8 in
            Bytes.set_int64_le b 0 v;
            Bytes.to_string b
          in
          let ok = ref true in
          for i = 0 to n - 1 do
            let c = coord () in
            (match dtype with
            | Dtype.F32 ->
                let b = Bytes.create 4 in
                Bytes.set_int32_le b 0
                  (Int32.bits_of_float (Tensor.read packed c));
                put i (Bytes.to_string b)
            | Dtype.F64 ->
                put i (le64 (Int64.bits_of_float (Tensor.read packed c)))
            | Dtype.I64 -> (
                match
                  Err.payload
                    (Tensor.read_i64_at6 packed (fun a ->
                         Vec6.get c a |> Dim.to_int))
                with
                | Ok v -> put i (le64 v)
                | Error _ -> ok := false)
            | Dtype.I32 ->
                (* Widened exactly; stored back at four bytes. *)
                let b = Bytes.create 4 in
                Bytes.set_int32_le b 0 (Int32.of_float (Tensor.read packed c));
                put i (Bytes.to_string b)
            | Dtype.BOOL -> (
                match
                  Err.payload
                    (Tensor.read_bool_at6 packed (fun a ->
                         Vec6.get c a |> Dim.to_int))
                with
                | Ok v -> put i (if v then "\001" else "\000")
                | Error _ -> ok := false)
            | _ -> ok := false);
            (* Odometer over the used axes, last fastest. *)
            let d = ref (rank - 1) and carry = ref true in
            while !carry && !d >= 0 do
              index.(!d) <- index.(!d) + 1;
              if index.(!d) = List.nth dims !d then begin
                index.(!d) <- 0;
                decr d
              end
              else carry := false
            done
          done;
          if not !ok then
            Error (`Native_layout "payload format does not match its dtype")
          else
            Ok
              {
                F.Logical.data = out;
                dtype;
                shape = List.map Int64.of_int dims;
              }

(* --- cases ---------------------------------------------------------------- *)

let tensor_error name e = `Logical_tensor (name, e)

(* A [.pt] map as logical tensors in the contract's order, or the reason it is
   not the set the descriptor names. *)
let load_tensors (b : Pt2_fixture_unix.Bundle.t) path ~names =
  let* pt = Pt2_archive.load_pt_tensor_map (Filename.concat b.dir path) in
  let listed = List.sort String.compare (List.map fst pt) in
  if listed <> List.sort String.compare names then
    Err.return (Error (List.map fst pt))
  else
    let* ordered =
      Err.List.map
        (fun name ->
          let tensor = List.assoc name pt in
          let+ logical =
            F.Logical.of_pt2 tensor
            |> Err.map_error (fun (e : F.Fault.Logical_fault.t) ->
                tensor_error name e)
          in
          (name, tensor, logical))
        names
    in
    Err.return (Ok ordered)

let digest_of named =
  match F.Tensor_digest.digest named with
  | Ok d -> Err.return d
  | Error (`Digest_name n) -> Err.fail (`Digest_name n)

let spec_matches (spec : F.Contract.Tensor_spec.t) (l : F.Logical.t) =
  Dtype.equal spec.dtype l.dtype && List.equal Int64.equal spec.shape l.shape

let failed_case id error =
  {
    Report.id;
    error = Some error;
    inputs_digest_ok = false;
    outputs = [];
    outputs_digest_ok = false;
  }

let run_case ~on_empty_caches ~dots archive (contract : F.Contract.t)
    (b : Pt2_fixture_unix.Bundle.t) (c : F.Cases.Case.t) =
  let input_names = c.inputs and output_names = c.outputs in
  let path role = Printf.sprintf "cases/%s/%s.pt" c.id role in
  let* inputs = load_tensors b (path "inputs") ~names:input_names in
  let* outputs = load_tensors b (path "outputs") ~names:output_names in
  match (inputs, outputs) with
  | Error got, _ ->
      Err.return
        ( failed_case c.id
            (Printf.sprintf "inputs.pt holds [%s], expected [%s]"
               (String.concat "; " got)
               (String.concat "; " input_names)),
          None )
  | _, Error got ->
      Err.return
        ( failed_case c.id
            (Printf.sprintf "outputs.pt holds [%s], expected [%s]"
               (String.concat "; " got)
               (String.concat "; " output_names)),
          None )
  | Ok inputs, Ok outputs -> (
      let* in_digest = digest_of (List.map (fun (n, _, l) -> (n, l)) inputs) in
      let* out_digest =
        digest_of (List.map (fun (n, _, l) -> (n, l)) outputs)
      in
      let inputs_ok = Pt2_sha256.Digest.equal in_digest c.inputs_sha256 in
      let outputs_ok = Pt2_sha256.Digest.equal out_digest c.outputs_sha256 in
      let mismatched_specs =
        List.filter_map
          (fun ((spec : F.Contract.Tensor_spec.t), (_, _, l)) ->
            if spec_matches spec l then None else Some spec.name)
          (List.combine contract.inputs inputs)
      in
      if mismatched_specs <> [] then
        Err.return
          ( {
              (failed_case c.id
                 ("input tensors differ from the contract: "
                 ^ String.concat ", " mismatched_specs))
              with
              inputs_digest_ok = inputs_ok;
              outputs_digest_ok = outputs_ok;
            },
            None )
      else
        match
          Err.payload
            (Native_interp.run_named ~empty_caches:on_empty_caches
               ~dot_accumulation:dots archive
               ~inputs:(List.map (fun (n, t, _) -> (n, t)) inputs))
        with
        | Error e ->
            (* The engine refused the graph or the call: a measured outcome of
               the whole artifact, not of this case alone. *)
            let text = Fmt.str "%a" Native_interp.pp_error e in
            Err.return
              ( {
                  (failed_case c.id text) with
                  inputs_digest_ok = inputs_ok;
                  outputs_digest_ok = outputs_ok;
                },
                Some text )
        | Ok actual when List.length actual <> List.length outputs ->
            Err.return
              ( {
                  (failed_case c.id
                     (Printf.sprintf
                        "graph returned %d outputs, contract says %d"
                        (List.length actual) (List.length outputs)))
                  with
                  inputs_digest_ok = inputs_ok;
                  outputs_digest_ok = outputs_ok;
                },
                None )
        | Ok actual ->
            let results =
              List.map2
                (fun ((name, _, expected), (spec : F.Contract.Tensor_spec.t))
                     native ->
                  match to_logical native ~rank:(List.length spec.shape) with
                  | Ok actual ->
                      F.Compare.tensor ~atol:contract.atol ~rtol:contract.rtol
                        ~name ~expected ~actual
                  | Error (`Native_layout why) ->
                      {
                        F.Compare.elements = F.Logical.numel expected;
                        first = [];
                        max_abs_error = 0.;
                        max_rel_error = 0.;
                        mismatches = 0L;
                        name = name ^ " (" ^ why ^ ")";
                        verdict =
                          F.Compare.Shape_differs
                            { actual = []; expected = expected.shape };
                      })
                (List.combine outputs contract.outputs)
                actual
            in
            Err.return
              ( {
                  Report.id = c.id;
                  error = None;
                  inputs_digest_ok = inputs_ok;
                  outputs = results;
                  outputs_digest_ok = outputs_ok;
                },
                None ))

(* One line per empty source, so the report records what was rewritten and
   where. Every case runs the same graph, so the first case's account stands
   for all of them. *)
let describe_empty_caches into (r : Native_interp.Empty_cache_report.t) =
  if !into = [] then
    into :=
      List.map
        (fun (s : Native_interp.Empty_cache_report.source) ->
          Printf.sprintf "empty-cache %s: dropped clones [%s]; cat operands %s"
            s.ssa
            (String.concat "; " s.clones)
            (match s.cats with
            | [] -> "none (unread)"
            | cats ->
                String.concat ", "
                  (List.map
                     (fun (c : Native_interp.Empty_cache_report.cat) ->
                       Printf.sprintf "%s at %s" c.cat
                         (String.concat ","
                            (List.map
                               (fun p ->
                                 string_of_int
                                   (p
                                     : Native_interp.Empty_cache_report.Operand
                                       .t
                                     :> int))
                               c.removed)))
                     cats)))
        r.sources

let replay ?(dots = Direct.Binary64) ~consumer (f : Pt2_fixture_unix.Fixture.t)
    =
  let b = f.bundle in
  let read name = Pt2_fixture_unix.Bundle.read_member b name in
  let* contract_text = read "contract.json" in
  let* contract = F.Contract.of_string contract_text in
  let* cases_text = read "cases.json" in
  let* cases = F.Cases.of_string cases_text in
  let* () = F.Cases.check contract cases in
  let hex = Pt2_sha256.Digest.to_hex in
  let pins =
    [
      ("archive", hex b.entry.archive.sha256);
      ("manifest", hex b.entry.manifest.sha256);
      ("graph", hex b.entry.graph_sha256);
      ("contract", hex b.entry.contract_sha256);
      ("map", hex b.entry.map_sha256);
    ]
    @ List.map
        (fun (s : Pt2_checkpoint_map.Document.Source.t) ->
          ("source:" ^ s.pin.name, hex s.pin.sha256))
        f.document.checkpoint_files
  in
  let normalizations = ref [] in
  let on_empty_caches = describe_empty_caches normalizations in
  let base status refusal cases =
    {
      Report.artifact_id = contract.artifact_id;
      atol = contract.atol;
      backend = backend_of dots;
      cases;
      consumer;
      normalizations = !normalizations;
      pins;
      refusal;
      rtol = contract.rtol;
      status;
    }
  in
  if contract.dynamic then
    Err.return
      (base Report.Refused
         (Some "dynamic-shape contract: no accepted component for the history")
         [])
  else
    let* results =
      Err.List.map
        (run_case ~on_empty_caches ~dots f.archive contract b)
        cases.cases
    in
    let cases = List.map fst results in
    match List.find_map snd results with
    | Some why -> Err.return (base Report.Refused (Some why) cases)
    | None -> Err.return (base (Report.status_of_cases cases) None cases)
