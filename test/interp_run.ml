(* Run a .pt2 model through the interpreter on every sample image and compare the
   local top-5 against the reference results.json shipped in the release zip.
   Driven by interp_cram (see `make pt2.runtest`) and `make inference`.
   argv: <model.pt2> <images_dir> <synsets.txt> <metadata.txt> <results.json> [--cram]

   Archive-open time, and each image's load/inference time, are printed to
   stderr (not stdout), since they are non-deterministic and would otherwise
   break the cram tests' exact-match comparison.

   --cram selects cram mode: probabilities differ in their low-order digits
   across systems, so no fixed rounding is exact-match safe. When the local
   top-5 ranking agrees with the reference, probabilities are dropped from
   the output entirely; only the ranking is compared. On a mismatch, or
   without --cram, probabilities print at natural precision for inspection. *)

module SMap = Map.Make (String)

type error =
  [ Pt2_archive.error
  | Interp.error
  | `Results_decode of string
  | `No_reference of string ]

let pp_error ppf : [< error ] -> unit = function
  | #Pt2_archive.error as e -> Pt2_archive.pp_error ppf e
  | #Interp.error as e -> Interp.pp_error ppf e
  | `Results_decode msg -> Format.fprintf ppf "results.json: %s" msg
  | `No_reference key -> Format.fprintf ppf "no reference for %S" key

let read_lines path =
  In_channel.with_open_text path (fun ic ->
      let rec go acc =
        match In_channel.input_line ic with
        | Some l -> go (l :: acc)
        | None -> List.rev acc
      in
      go [])

let names_of_metadata path =
  let tbl = Hashtbl.create 4096 in
  List.iter
    (fun line ->
      Option.iter
        (fun i ->
          Hashtbl.replace tbl (String.sub line 0 i)
            (String.sub line (i + 1) (String.length line - i - 1)))
        (String.index_opt line '\t'))
    (read_lines path);
  tbl

(* --- reference results.json: { "images/<f>.pt": [ {rank,class_index,...} ] } --- *)

(* One reference prediction, mirroring the results.json schema. *)
type pred = {
  rank : int;
  class_index : int;
  label : string;
  probability : float;
}

let pred_jsont =
  Jsont.Object.map ~kind:"pred" (fun rank class_index label probability ->
      { rank; class_index; label; probability })
  |> Jsont.Object.mem "rank" Jsont.int
  |> Jsont.Object.mem "class_index" Jsont.int
  |> Jsont.Object.mem "label" Jsont.string
  |> Jsont.Object.mem "probability" Jsont.number
  |> Jsont.Object.finish

let results_jsont = Jsont.Object.as_string_map (Jsont.list pred_jsont)

let decode_results path =
  let s = In_channel.with_open_bin path In_channel.input_all in
  match Jsont_bytesrw.decode_string results_jsont s with
  | Ok m -> Err.return m
  | Error e -> Err.fail (`Results_decode e)

type mode = Natural | Cram

(* Prints a (local, reference) probability pair for a top-5 entry that
   matches the reference ranking. In cram mode the pair is dropped
   entirely, since probabilities computed on different systems can differ
   in their low-order digits and would otherwise break the cram tests'
   exact-match comparison; the ranking match is the only assertion there. *)
let pp_match mode oc (local, refp) =
  match mode with
  | Cram -> ()
  | Natural -> Printf.fprintf oc " (%g, ref %g)" local refp

(* Runs [f], then prints its wall-clock time under [label] to stderr. *)
let timed label f =
  let t0 = Unix.gettimeofday () in
  let result = f () in
  Printf.eprintf "%s: %.1f ms\n%!" label ((Unix.gettimeofday () -. t0) *. 1000.);
  result

let main () =
  let open Err.Syntax in
  let pt2 = Sys.argv.(1) and images_dir = Sys.argv.(2) in
  let synsets_path = Sys.argv.(3) and metadata_path = Sys.argv.(4) in
  let results_path = Sys.argv.(5) in
  let mode =
    if Array.length Sys.argv > 6 && Sys.argv.(6) = "--cram" then Cram
    else Natural
  in
  let* archive =
    timed "pt2 open" (fun () -> Pt2_archive.open_pt2 pt2)
    |> Err.map_error (fun e -> (e :> error))
  in
  let synsets = Array.of_list (read_lines synsets_path) in
  let names = names_of_metadata metadata_path in
  let label idx =
    let s = synsets.(idx) in
    s ^ " " ^ Option.value (Hashtbl.find_opt names s) ~default:"?"
  in
  let* results = decode_results results_path in
  let files =
    Sys.readdir images_dir |> Array.to_list
    |> List.filter (fun f -> Filename.check_suffix f ".pt")
    |> List.sort String.compare
  in
  let* () =
    Err.List.fold_left
      (fun () f ->
        let key = "images/" ^ f in
        let* image =
          timed (key ^ " load") (fun () ->
              Pt2_archive.load_pt (Filename.concat images_dir f))
          |> Err.map_error (fun e -> (e :> error))
        in
        let* logits =
          timed (key ^ " infer") (fun () -> Interp.run archive image)
          |> Err.map_error (fun e -> (e :> error))
        in
        let* local =
          Interp.top_predictions logits 5
          |> Err.map_error (fun e -> (e :> error))
        in
        let* expected =
          match SMap.find_opt key results with
          | Some ps -> Err.return ps
          | None -> Err.fail (`No_reference key)
        in
        Printf.printf "=== %s ===\n" key;
        let matches =
          List.map fst local = List.map (fun p -> p.class_index) expected
        in
        if matches then
          List.iteri
            (fun i (idx, prob) ->
              let p = List.nth expected i in
              Printf.printf "%d: %s%a\n" (i + 1) (label idx) (pp_match mode)
                (prob, p.probability))
            local
        else begin
          print_endline "local:";
          List.iteri
            (fun i (idx, prob) ->
              Printf.printf "%d: %s (%g)\n" (i + 1) (label idx) prob)
            local;
          print_endline "reference:";
          List.iter
            (fun p ->
              Printf.printf "%d: %s (%g)\n" p.rank p.label p.probability)
            expected
        end;
        Err.return ())
      () files
  in
  Err.return ()

let () =
  match main () with
  | Ok () -> ()
  | Error e ->
      Format.eprintf "%a@." (Err.Error.pp pp_error) e;
      exit 1
