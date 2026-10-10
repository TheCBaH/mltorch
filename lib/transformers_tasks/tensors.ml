open Err.Syntax
open Transformers_metadata.Json_util
open Spec
module D = Pt2_checkpoint_map.Dtype
module L = F.Logical

let load path descriptors =
  let* rows = array descriptors in
  let* names = Err.List.map (field "name") rows in
  let* () = unique names in
  let* actual = Pt2_archive.load_pt_tensor_map path in
  let actual_names = List.map fst actual in
  let* () = unique actual_names in
  let* () =
    require
      (List.sort String.compare names = List.sort String.compare actual_names)
      ("tensor names/order descriptor set: " ^ path)
  in
  Err.List.map
    (fun row ->
      let* name = field "name" row in
      let* dtype_name = field "dtype" row in
      let* dtype =
        Err.of_option (`Unknown_dtype dtype_name) (D.of_torch_name dtype_name)
      in
      let* () =
        require (List.mem dtype [ D.F32; D.I64; D.U8; D.BOOL ]) "task dtype"
      in
      let* shape = member "shape" row >>= array >>= Err.List.map integer in
      let* tensor =
        Err.of_option (`Missing_tensor name) (List.assoc_opt name actual)
      in
      let* logical =
        L.of_pt2 tensor |> Err.map_error (fun e -> `Logical_tensor (name, e))
      in
      let* () =
        require
          (D.equal dtype logical.dtype && shape = logical.shape)
          ("tensor dtype/shape: " ^ name)
      in
      let* expected_sha = field "sha256" row >>= digest in
      let* () =
        F.Check.digest
          (F.Fault.Member (path ^ ":" ^ name))
          (Pt2_sha256.bigstring logical.data)
          expected_sha
      in
      Ok (name, logical))
    rows

let case_files bundle case =
  let* id = field "id" case in
  let* files = member "files" case >>= members in
  let* () = require (files <> []) "case has no tensor files" in
  let* files =
    Err.List.map
      (fun (name, descriptors) ->
        let* () =
          require
            (U.Targz.safe_name name
            && String.starts_with ~prefix:("cases/" ^ id ^ "/") name
            && Smap.mem name bundle.Reference.Bundle.inventory)
            ("case tensor path: " ^ name)
        in
        let+ descriptors = member "tensors" descriptors in
        (name, descriptors))
      files
  in
  let* raw = member "raw" case >>= array in
  let* () = require (raw <> []) "case has no raw inputs" in
  let* () =
    Err.List.iter
      (fun row ->
        let* name = field "path" row in
        let* actual =
          Err.of_option (`Member_missing name)
            (Smap.find_opt name bundle.inventory)
        in
        let* sha = field "sha256" row in
        let* size = member "size" row >>= integer in
        require
          (sha = Pt2_sha256.Digest.to_hex actual.sha256 && size = actual.size)
          ("raw input pin: " ^ name))
      raw
  in
  Ok files

let verify_case bundle case =
  let* files = case_files bundle case in
  Err.List.iter
    (fun (name, descriptors) ->
      let+ _ =
        load (Filename.concat bundle.Reference.Bundle.dir name) descriptors
      in
      ())
    files

let same_tensor_sets ~identity expected actual =
  let* () =
    require
      (List.map fst expected = List.map fst actual)
      (identity ^ " tensor order")
  in
  Err.List.iter
    (fun ((name, e), (actual_name, a)) ->
      let* () =
        require
          (name = actual_name && e.L.dtype = a.L.dtype && e.shape = a.shape)
          (identity ^ " tensor metadata: " ^ name)
      in
      F.Check.digest
        (F.Fault.Member (identity ^ ":" ^ name))
        (Pt2_sha256.bigstring a.data)
        (Pt2_sha256.bigstring e.data))
    (List.combine expected actual)

let original bundle relative descriptors =
  let* () =
    require
      (Smap.mem relative bundle.U.Bundle.manifest.members)
      ("undeclared published member " ^ relative)
  in
  load (Filename.concat bundle.dir relative) descriptors
