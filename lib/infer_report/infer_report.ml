(* See infer_report.mli for the contract.

   The split between [run] and [report] is the load-bearing structure here.
   [run] owns everything that touches the host -- argv, the filesystem, the
   archive, the clock -- and [report] owns the loop, the formatting and the
   verdict, with all of that injected. Without the split there is no way to
   observe "printed every sample, then failed strict" except by running a real
   model against a deliberately wrong results.json. *)

module SMap = Map.Make (String)

type mode = Natural | Cram

module Paths = struct
  type t = {
    pt2 : string;
    images_dir : string;
    synsets : string;
    metadata : string;
    results : string;
  }
end

module Options = struct
  type t = { mode : mode; strict : bool }
end

module Mismatch = struct
  type t = { sample : string; local : int list; reference : int list }
end

module Prediction = struct
  type t = {
    rank : int;
    class_index : int;
    label : string;
    probability : float;
  }
end

type 'eval error =
  [ Pt2_archive.error
  | `Results_decode of string
  | `No_reference of string
  | `Mismatch of Mismatch.t
  | `Eval of 'eval ]

let pp_error pp_eval ppf : 'eval error -> unit = function
  | #Pt2_archive.error as e -> Pt2_archive.pp_error ppf e
  | `Results_decode msg -> Fmt.pf ppf "results.json: %s" msg
  | `No_reference key -> Fmt.pf ppf "no reference for %S" key
  | `Mismatch { Mismatch.sample; local; reference } ->
      let pp_indices = Fmt.(list ~sep:(any ", ") int) in
      Fmt.pf ppf "@[<h>%s: top-%d [%a] does not match reference [%a]@]" sample
        (List.length local) pp_indices local pp_indices reference
  (* No prefix: an evaluator failure has to read exactly as it did before this
     library existed, since that is what the interpreter crams' stderr shows. *)
  | `Eval e -> pp_eval ppf e

(* --- label files --- *)

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

let prediction_jsont =
  Jsont.Object.map ~kind:"pred" (fun rank class_index label probability ->
      { Prediction.rank; class_index; label; probability })
  |> Jsont.Object.mem "rank" Jsont.int
  |> Jsont.Object.mem "class_index" Jsont.int
  |> Jsont.Object.mem "label" Jsont.string
  |> Jsont.Object.mem "probability" Jsont.number
  |> Jsont.Object.finish

let results_jsont = Jsont.Object.as_string_map (Jsont.list prediction_jsont)

let decode_results path =
  let s = In_channel.with_open_bin path In_channel.input_all in
  match Jsont_bytesrw.decode_string results_jsont s with
  | Ok m -> Err.return m
  | Error e -> Err.fail (`Results_decode e)

(* --- command line --- *)

let usage =
  "usage: <model.pt2> <images_dir> <synsets.txt> <metadata.txt> <results.json> \
   [--cram] [--strict]"

let is_flag s = String.length s > 0 && s.[0] = '-'

(* The five paths first, then only flags. A flag in a positional slot and a
   sixth path are both rejected here rather than ignored: silently dropping a
   trailing argument is how a caller ends up believing it passed --strict. *)
let rec take_paths n acc args =
  if n = 0 then Ok (List.rev acc, args)
  else
    match args with
    | [] -> Error usage
    | a :: _ when is_flag a -> Error usage
    | a :: rest -> take_paths (n - 1) (a :: acc) rest

let rec take_flags mode strict = function
  | [] -> Ok { Options.mode; strict }
  | "--cram" :: rest -> take_flags Cram strict rest
  | "--strict" :: rest -> take_flags mode true rest
  | _ -> Error usage

let parse_argv argv =
  match Array.to_list argv with
  | [] -> Error usage
  | _program :: args -> (
      match take_paths 5 [] args with
      | Error _ as e -> e
      | Ok ([ pt2; images_dir; synsets; metadata; results ], rest) -> (
          match take_flags Natural false rest with
          | Error _ as e -> e
          | Ok options ->
              Ok ({ Paths.pt2; images_dir; synsets; metadata; results }, options)
          )
      | Ok _ -> Error usage)

(* --- comparison --- *)

let compare_ranking ~sample ~local ~reference =
  let local = List.map fst local in
  if local = reference then None else Some { Mismatch.sample; local; reference }

let strict_verdict = function [] -> Ok () | first :: _ -> Error first

(* --- the report --- *)

(* The (local, reference) probability pair for an entry whose ranking matched.
   Dropped in cram mode: probabilities computed on different systems differ in
   their low-order digits, and no fixed rounding is exact-match safe. *)
let pp_match mode oc (local, refp) =
  match mode with
  | Cram -> ()
  | Natural -> Printf.fprintf oc " (%g, ref %g)" local refp

let print_sample ~label ~mode ~sample ~local ~expected ~matched =
  Printf.printf "=== %s ===\n" sample;
  if matched then
    List.iteri
      (fun i (idx, prob) ->
        let p = List.nth expected i in
        Printf.printf "%d: %s%a\n" (i + 1) (label idx) (pp_match mode)
          (prob, p.Prediction.probability))
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
        Printf.printf "%d: %s (%g)\n" p.Prediction.rank p.Prediction.label
          p.Prediction.probability)
      expected
  end

let report ~label ~reference ~samples ~infer (options : Options.t) =
  let open Err.Syntax in
  let* mismatches =
    Err.List.fold_left
      (fun mismatches sample ->
        let* local = infer sample in
        let* expected =
          match reference sample with
          | Some ps -> Err.return ps
          | None -> Err.fail (`No_reference sample)
        in
        let mismatch =
          compare_ranking ~sample ~local
            ~reference:(List.map (fun p -> p.Prediction.class_index) expected)
        in
        print_sample ~label ~mode:options.mode ~sample ~local ~expected
          ~matched:(Option.is_none mismatch);
        Err.return
          (match mismatch with None -> mismatches | Some m -> m :: mismatches))
      [] samples
  in
  (* Every sample has already been reported by here, so a strict failure adds a
     verdict to a complete report rather than truncating it. *)
  if not options.strict then Err.return ()
  else
    match strict_verdict (List.rev mismatches) with
    | Ok () -> Err.return ()
    | Error m -> Err.fail (`Mismatch m)

(* --- the host shell --- *)

(* Runs [f], then prints its wall-clock time under [label] to stderr. *)
let timed now label f =
  let t0 = now () in
  let result = f () in
  Printf.eprintf "%s: %.1f ms\n%!" label ((now () -. t0) *. 1000.);
  result

let run ~now ~infer (paths : Paths.t) options =
  let open Err.Syntax in
  let* archive =
    timed now "pt2 open" (fun () -> Pt2_archive.open_pt2 paths.pt2)
    |> Err.map_error (fun e -> (e : Pt2_archive.error :> _ error))
  in
  let synsets = Array.of_list (read_lines paths.synsets) in
  let names = names_of_metadata paths.metadata in
  let label idx =
    let s = synsets.(idx) in
    s ^ " " ^ Option.value (Hashtbl.find_opt names s) ~default:"?"
  in
  let* results = decode_results paths.results in
  let samples =
    Sys.readdir paths.images_dir
    |> Array.to_list
    |> List.filter (fun f -> Filename.check_suffix f ".pt")
    |> List.sort String.compare
    |> List.map (fun f -> "images/" ^ f)
  in
  (* The one place the two kinds of per-sample failure are told apart. A
     [load_pt] failure widens straight into the row and keeps its own tag; only
     the evaluator call nests under [`Eval], and only it marks a boundary. *)
  let load_and_infer sample =
    let path = Filename.concat paths.images_dir (Filename.basename sample) in
    let* image =
      timed now (sample ^ " load") (fun () -> Pt2_archive.load_pt path)
      |> Err.map_error (fun e -> (e : Pt2_archive.error :> _ error))
    in
    timed now (sample ^ " infer") (fun () -> infer archive image)
    |> Err.map_error ~pos:__POS__ (fun e -> `Eval e)
  in
  report ~label
    ~reference:(fun sample -> SMap.find_opt sample results)
    ~samples ~infer:load_and_infer options
