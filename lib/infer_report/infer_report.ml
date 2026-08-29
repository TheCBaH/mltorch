(* See infer_report.mli for the contract.

   The split between [run] and [report] is the load-bearing structure here.
   [run] owns everything that touches the host -- argv, the filesystem, the
   archive, the clock -- and [report] owns the loop, the formatting and the
   verdict, with all of that injected. Without the split there is no way to
   observe "printed every sample, then failed strict" except by running a real
   model against a deliberately wrong results.json. *)

module SMap = Map.Make (String)

type mode = Cram | Natural

module Paths = struct
  type t = {
    pt2 : string;
    images_dir : string;
    synsets : string;
    metadata : string;
    results : string;
    inputs : string option;
    expected : string option;
    outputs : string option;
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
  [ `Eval of 'eval
  | `Expected_decode of string
  | `Mismatch of Mismatch.t
  | `No_reference of string
  | Pt2_archive.error
  | `Results_decode of string ]

let pp_error pp_eval ppf : 'eval error -> unit = function
  (* No prefix: an evaluator failure has to read exactly as it did before this
     library existed, since that is what the interpreter crams' stderr shows. *)
  | `Eval e -> pp_eval ppf e
  | `Expected_decode msg -> Fmt.pf ppf "expected.json: %s" msg
  | `Mismatch { Mismatch.sample; local; reference } ->
      let pp_indices = Fmt.(list ~sep:(any ", ") int) in
      Fmt.pf ppf "@[<h>%s: top-%d [%a] does not match reference [%a]@]" sample
        (List.length local) pp_indices local pp_indices reference
  | `No_reference key -> Fmt.pf ppf "no reference for %S" key
  | #Pt2_archive.error as e -> Pt2_archive.pp_error ppf e
  | `Results_decode msg -> Fmt.pf ppf "results.json: %s" msg

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
  Jsont.Object.map ~kind:"pred" (fun class_index label probability rank ->
      { Prediction.rank; class_index; label; probability })
  |> Jsont.Object.mem "class_index" Jsont.int
  |> Jsont.Object.mem "label" Jsont.string
  |> Jsont.Object.mem "probability" Jsont.number
  |> Jsont.Object.mem "rank" Jsont.int
  |> Jsont.Object.finish

let results_jsont = Jsont.Object.as_string_map (Jsont.list prediction_jsont)

module Expected = struct
  type t = { top5 : int list; logits : float list }
end

let expected_jsont =
  Jsont.Object.map ~kind:"expected" (fun logits top5 ->
      { Expected.top5; logits })
  |> Jsont.Object.mem "logits" (Jsont.list Jsont.number)
  |> Jsont.Object.mem "top5" (Jsont.list Jsont.int)
  |> Jsont.Object.finish

let expected_map_jsont = Jsont.Object.as_string_map expected_jsont

let decode_results path =
  let s = In_channel.with_open_bin path In_channel.input_all in
  match Jsont_bytesrw.decode_string results_jsont s with
  | Error e -> Err.fail (`Results_decode e)
  | Ok m -> Err.return m

let decode_expected path =
  let s = In_channel.with_open_bin path In_channel.input_all in
  match Jsont_bytesrw.decode_string expected_map_jsont s with
  | Error e -> Err.fail (`Expected_decode e)
  | Ok m -> Err.return m

(* --- command line --- *)

let usage =
  "usage: <model.pt2> <inputs.pt> <expected.json> <outputs.pt> [--cram] \
   [--strict]"

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
      match take_paths 4 [] args with
      | Error _ as e -> e
      | Ok ([ pt2; inputs; expected; outputs ], rest) -> (
          match take_flags Natural false rest with
          | Error _ as e -> e
          | Ok options ->
              Ok
                ( {
                    Paths.pt2;
                    images_dir = "";
                    synsets = "";
                    metadata = "";
                    results = "";
                    inputs = Some inputs;
                    expected = Some expected;
                    outputs = Some outputs;
                  },
                  options ))
      | Ok _ -> Error usage)

(* --- comparison --- *)

let compare_ranking ~sample ~local ~reference =
  let local = List.map fst local in
  if local = reference then None else Some { Mismatch.sample; local; reference }

let strict_verdict = function [] -> Ok () | first :: _ -> Error first

(* Producer input maps are saved after timm preprocessing and before the model
   batch dimension is added.  Creating the leading size/stride is a view over
   the same storage, so it neither copies nor changes the external tensor-map
   contract. *)
let with_batch (tensor : Pt2_tensor.t) =
  let stride = Pt2_tensor.numel tensor in
  {
    tensor with
    Pt2_tensor.sizes = 1 :: tensor.sizes;
    strides = stride :: tensor.strides;
  }

let exact_keys what left right =
  let left = List.map fst left and right = List.map fst right in
  if left = right then Err.return ()
  else
    Err.fail
      (`Results_decode
         (Fmt.str "%s keys do not agree: inputs=[%s] oracle=[%s]" what
            (String.concat ", " left) (String.concat ", " right)))

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
    | Error m -> Err.fail (`Mismatch m)
    | Ok () -> Err.return ()

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
  match (paths.inputs, paths.expected, paths.outputs) with
  | Some inputs_path, Some expected_path, Some outputs_path ->
      let* inputs =
        timed now "inputs load" (fun () ->
            Pt2_archive.load_pt_tensor_map inputs_path)
        |> Err.map_error (fun e -> (e : Pt2_archive.error :> _ error))
      in
      let* outputs =
        timed now "outputs load" (fun () ->
            Pt2_archive.load_pt_tensor_map outputs_path)
        |> Err.map_error (fun e -> (e : Pt2_archive.error :> _ error))
      in
      let* expected = decode_expected expected_path in
      let expected_pairs = SMap.bindings expected in
      let* () = exact_keys "inputs/outputs" inputs outputs in
      let* () = exact_keys "inputs/expected" inputs expected_pairs in
      let reference sample =
        match SMap.find_opt sample expected with
        | None -> None
        | Some e ->
            Some
              (List.mapi
                 (fun rank class_index ->
                   {
                     Prediction.rank = rank + 1;
                     class_index;
                     label = string_of_int class_index;
                     probability =
                       Option.value
                         (List.nth_opt e.Expected.logits rank)
                         ~default:nan;
                   })
                 e.Expected.top5)
      in
      let load_and_infer sample =
        match List.assoc_opt sample inputs with
        | None -> Err.fail (`No_reference sample)
        | Some input ->
            timed now (sample ^ " infer") (fun () ->
                infer archive (with_batch input))
            |> Err.map_error ~pos:__POS__ (fun e -> `Eval e)
      in
      report
        ~label:(fun i -> string_of_int i)
        ~reference ~samples:(List.map fst inputs) ~infer:load_and_infer options
  | _ ->
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
        let path =
          Filename.concat paths.images_dir (Filename.basename sample)
        in
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
