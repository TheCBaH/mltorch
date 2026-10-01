(* The arena evaluation's report, computed from its JSONL rows alone: coverage
   per dialect, reference outcomes, a per-dialect strategy leaderboard, paired
   cross-dialect sizes, the hardest unresolved problems and planning cost.
   Every figure states its denominator. *)

module J = Arena_eval_json

let get_s = J.string_field
let get_i64 name j = J.get J.to_int64 name j |> Option.value ~default:0L
let get_f name j = J.get J.to_float name j |> Option.value ~default:0.
let get_i name j = J.get J.to_int name j |> Option.value ~default:0
let get_b name j = J.get J.to_bool name j |> Option.value ~default:false
let is t j = get_s "type" j = Some t
let key j = (get_s "model" j, get_s "dialect" j, get_s "kind" j)

module Coverage = struct
  type t = {
    dialect : string;
    attempted : int;
    evaluated : int;
    refused : (string * int) list;  (** "stage/reason" and count. *)
    prerequisite : int;
    failed : (string * string * string) list;  (** Model, stage, diagnostic. *)
  }
end

type t = {
  expected : int;
  dialects : string list;
  rows : Jsont.json list;
  models : int;
  coverage : Coverage.t list;
}

let count_by f xs =
  List.fold_left
    (fun acc x ->
      let k = f x in
      match List.assoc_opt k acc with
      | Some n -> (k, n + 1) :: List.remove_assoc k acc
      | None -> (k, 1) :: acc)
    [] xs
  |> List.sort compare

let coverage rows dialect =
  let ds =
    List.filter
      (fun j -> is "dialect" j && get_s "dialect" j = Some dialect)
      rows
  in
  let with_status s = List.filter (fun j -> get_s "status" j = Some s) ds in
  let opt name j = Option.value (get_s name j) ~default:"?" in
  {
    Coverage.dialect;
    attempted = List.length ds;
    evaluated = List.length (with_status "evaluated");
    refused =
      count_by
        (fun j -> opt "stage" j ^ "/" ^ opt "reason" j)
        (with_status "refused");
    prerequisite = List.length (with_status "prerequisite");
    failed =
      List.map
        (fun j -> (opt "model" j, opt "stage" j, opt "diagnostic" j))
        (with_status "failed");
  }

let of_rows ~expected ~dialects rows =
  {
    expected;
    dialects;
    rows;
    models = List.length (List.filter (is "model_done") rows);
    coverage = List.map (coverage rows) dialects;
  }

(* Every model attempted in every dialect; Native evaluated for all; nothing
   failed and nothing blocked by a failed import. A classified Native4D
   refusal is a result. *)
let ok t =
  t.models = t.expected
  && List.for_all
       (fun (c : Coverage.t) ->
         c.attempted = t.expected && c.failed = [] && c.prerequisite = 0
         && (c.dialect <> "native" || c.evaluated = t.expected))
       t.coverage

let coverage_rows t =
  List.map
    (fun (c : Coverage.t) ->
      J.obj
        [
          ("type", J.str "coverage");
          ("dialect", J.str c.dialect);
          ("expected", J.int t.expected);
          ("attempted", J.int c.attempted);
          ("evaluated", J.int c.evaluated);
          ("refused", J.int (List.fold_left (fun n (_, k) -> n + k) 0 c.refused));
          ("prerequisite", J.int c.prerequisite);
          ("failed", J.int (List.length c.failed));
        ])
    t.coverage

(* --- markdown ------------------------------------------------------------- *)

let pct n d =
  if d = 0 then "-" else Printf.sprintf "%.1f%%" (100. *. float n /. float d)

let table buf header rows =
  let line cells =
    Buffer.add_string buf ("| " ^ String.concat " | " cells ^ " |\n")
  in
  line header;
  line (List.map (fun _ -> "---") header);
  List.iter line rows;
  Buffer.add_char buf '\n'

let section buf title = Buffer.add_string buf ("\n## " ^ title ^ "\n\n")
let mib b = Printf.sprintf "%.2f" (Int64.to_float b /. 1048576.)

let header_section buf t =
  match List.find_opt (is "header") t.rows with
  | None -> ()
  | Some h ->
      let config = J.field "config" h and revisions = J.field "revisions" h in
      let s f name = Option.bind f (get_s name) |> Option.value ~default:"?" in
      let list name =
        match Option.bind config (J.field name) with
        | Some l ->
            Option.value (J.to_list l) ~default:[]
            |> List.filter_map J.to_string
            |> String.concat ","
        | None -> "?"
      in
      let reference name =
        Option.bind config (J.field "reference")
        |> Option.map (get_i64 name)
        |> Option.fold ~none:"?" ~some:Int64.to_string
      in
      Buffer.add_string buf
        (Printf.sprintf
           "Repository %s, models %s. Normalization: %s; retain %s. Order \
            search iterations %s, seeds %s, %d timing sample(s). Reference \
            limits: %s states, depth %s, per dialect and kind.\n"
           (s revisions "repo") (s revisions "models")
           (s config "normalization") (s config "retain") (list "iterations")
           (list "seeds")
           (Option.fold ~none:1 ~some:(get_i "repeats") config)
           (reference "max_states") (reference "max_depth"))

let coverage_section buf t =
  section buf "Coverage";
  table buf
    [ "dialect"; "attempted"; "evaluated"; "refused"; "prerequisite"; "failed" ]
    (List.map
       (fun (c : Coverage.t) ->
         [
           c.dialect;
           Printf.sprintf "%d/%d" c.attempted t.expected;
           Printf.sprintf "%d/%d" c.evaluated t.expected;
           string_of_int (List.fold_left (fun n (_, k) -> n + k) 0 c.refused);
           string_of_int c.prerequisite;
           string_of_int (List.length c.failed);
         ])
       t.coverage);
  List.iter
    (fun (c : Coverage.t) ->
      List.iter
        (fun (why, n) ->
          Buffer.add_string buf
            (Printf.sprintf "- %s refused at %s: %d/%d\n" c.dialect why n
               t.expected))
        c.refused;
      List.iter
        (fun (m, stage, diag) ->
          Buffer.add_string buf
            (Printf.sprintf "- %s FAILED %s at %s: %s\n" c.dialect m stage diag))
        c.failed)
    t.coverage

let rows_of t ty dialect =
  List.filter (fun j -> is ty j && get_s "dialect" j = Some dialect) t.rows

let reference_section buf t =
  section buf "Reference minimum (per dialect and kind)";
  table buf
    [
      "dialect";
      "kinds";
      "optimal at live bound";
      "optimal above";
      "incomplete";
      "closed by a strategy";
      "states";
      "seconds";
    ]
    (List.map
       (fun d ->
         let refs = rows_of t "reference" d in
         let cmps = rows_of t "comparison" d in
         let n = List.length refs in
         let status s =
           List.length (List.filter (fun j -> get_s "status" j = Some s) refs)
         in
         let closed_by_strategy =
           List.length
             (List.filter
                (fun j ->
                  get_b "closed" j && get_s "proof" j <> Some "reference")
                cmps)
         in
         [
           d;
           string_of_int n;
           Printf.sprintf "%d (%s)"
             (status "optimal_live_bound")
             (pct (status "optimal_live_bound") n);
           string_of_int (status "optimal_above_live_bound");
           string_of_int (status "incomplete");
           string_of_int closed_by_strategy;
           Int64.to_string
             (List.fold_left
                (fun a j -> Int64.add a (get_i64 "states" j))
                0L refs);
           Printf.sprintf "%.2f"
             (List.fold_left (fun a j -> a +. get_f "seconds" j) 0. refs);
         ])
       t.dialects)

(* Per method and budget: how often it reaches the best pool any row found for
   the kind, the proven optimum, and the live bound, and its mean excess over
   the best observed (which is not an optimum). *)
let leaderboard_section buf t =
  List.iter
    (fun d ->
      let cmps = rows_of t "comparison" d in
      let best = Hashtbl.create 64 and closed = Hashtbl.create 64 in
      List.iter
        (fun j ->
          Hashtbl.replace best (key j) (get_i64 "best_observed" j);
          if get_b "closed" j then
            Hashtbl.replace closed (key j) (get_i64 "lower" j))
        cmps;
      let strat = rows_of t "strategy" d in
      let label j =
        match (J.get J.to_int64 "iterations" j, J.get J.to_int64 "seed" j) with
        | Some i, Some s ->
            Printf.sprintf "%s @%Ld/%Ld"
              (Option.value (get_s "method" j) ~default:"?")
              i s
        | _ -> Option.value (get_s "method" j) ~default:"?"
      in
      (* By method, then budget numerically, then seed. *)
      let order j =
        ( Option.value (get_s "method" j) ~default:"",
          J.get J.to_int64 "iterations" j,
          J.get J.to_int64 "seed" j )
      in
      let labels =
        List.sort_uniq (fun a b -> compare (order a) (order b)) strat
        |> List.map label
        |> List.fold_left
             (fun acc l -> if List.mem l acc then acc else l :: acc)
             []
        |> List.rev
      in
      section buf
        (Printf.sprintf "Leaderboard: %s (%d kinds)" d (List.length cmps));
      table buf
        [
          "method";
          "at best observed";
          "at proven optimum";
          "at live bound";
          "mean excess over best";
          "max excess";
          "seconds";
        ]
        (List.map
           (fun l ->
             let rs = List.filter (fun j -> label j = l) strat in
             let n = List.length rs in
             let excess j =
               let b =
                 Option.value (Hashtbl.find_opt best (key j)) ~default:0L
               in
               let p = get_i64 "pool" j in
               if Int64.equal b 0L then 0.
               else Int64.to_float (Int64.sub p b) /. Int64.to_float b
             in
             let at_best =
               List.length
                 (List.filter
                    (fun j ->
                      Some (get_i64 "pool" j) = Hashtbl.find_opt best (key j))
                    rs)
             in
             let proven =
               List.filter (fun j -> Hashtbl.mem closed (key j)) rs
             in
             let at_opt =
               List.length
                 (List.filter
                    (fun j ->
                      Some (get_i64 "pool" j) = Hashtbl.find_opt closed (key j))
                    proven)
             in
             let at_live =
               List.length (List.filter (fun j -> get_i64 "live_gap" j = 0L) rs)
             in
             let ex = List.map excess rs in
             [
               l;
               Printf.sprintf "%d/%d" at_best n;
               Printf.sprintf "%d/%d" at_opt (List.length proven);
               Printf.sprintf "%d/%d" at_live n;
               (if n = 0 then "-"
                else
                  Printf.sprintf "%.4f%%"
                    (100. *. List.fold_left ( +. ) 0. ex /. float n));
               Printf.sprintf "%.4f%%" (100. *. List.fold_left Float.max 0. ex);
               Printf.sprintf "%.3f"
                 (List.fold_left
                    (fun a j ->
                      a +. get_f "construct_s" j +. get_f "search_s" j
                      +. get_f "check_s" j)
                    0. rs);
             ])
           labels))
    t.dialects

(* Only models both dialects evaluated; each against its own bounds, since
   legalization changes the problem and is not an allocator gain. *)
(* --- per model ------------------------------------------------------------ *)

(* Checked: a byte total that overflows [int64] is reported as absent. Rows
   are in bytes already. *)
let add_bytes acc b =
  match acc with
  | Some acc when Int64.compare acc (Int64.sub Int64.max_int b) <= 0 ->
      Some (Int64.add acc b)
  | _ -> None

(* A strategy row's column: the method alone for a single pass, with its
   budget otherwise. *)
let column j =
  let m = Option.value (get_s "method" j) ~default:"?" in
  match (J.get J.to_int64 "iterations" j, J.get J.to_int64 "seed" j) with
  | Some i, Some s -> Printf.sprintf "%s@%Ld/%Ld" m i s
  | _ -> m

module Model_arena = struct
  type t = {
    model : string;
    dialect : string;
    kinds : int;
    lower : int64 option;
    upper : int64 option;
    proven : bool;
    placed : int64 option;  (** The exact-size production pool. *)
    pools : (string * int64 option) list;  (** Column, bytes; in row order. *)
  }
end

(* One entry per evaluated model and dialect, summed over its kinds: the
   minimum (proven, or bounded), and every strategy row's total. *)
let model_arenas t =
  let by_key = Hashtbl.create 256 and order = ref [] in
  let entry j =
    let k = (get_s "model" j, get_s "dialect" j) in
    match Hashtbl.find_opt by_key k with
    | Some e -> e
    | None ->
        let e =
          (ref 0, ref (Some 0L), ref (Some 0L), ref true, ref (Some 0L), ref [])
        in
        Hashtbl.replace by_key k e;
        order := k :: !order;
        e
  in
  List.iter
    (fun j ->
      if is "comparison" j then begin
        let kinds, lower, upper, proven, placed, _ = entry j in
        incr kinds;
        lower := add_bytes !lower (get_i64 "lower" j);
        upper := add_bytes !upper (get_i64 "upper" j);
        proven := !proven && get_b "closed" j;
        placed := add_bytes !placed (get_i64 "placed" j)
      end
      else if is "strategy" j then begin
        let _, _, _, _, _, pools = entry j in
        let c = column j in
        let prev =
          match List.assoc_opt c !pools with Some v -> v | None -> Some 0L
        in
        pools :=
          (c, add_bytes prev (get_i64 "pool" j)) :: List.remove_assoc c !pools
      end)
    t.rows;
  List.rev_map
    (fun ((m, d) as k) ->
      let kinds, lower, upper, proven, placed, pools = Hashtbl.find by_key k in
      {
        Model_arena.model = Option.value m ~default:"?";
        dialect = Option.value d ~default:"?";
        kinds = !kinds;
        lower = !lower;
        upper = !upper;
        proven = !proven;
        placed = !placed;
        pools = List.rev !pools;
      })
    !order

(* One row naming the columns, then one per evaluated model and dialect with
   its pools in that column order: every column is present in every row, since
   every kind runs the same matrix. *)
let model_rows t =
  let arenas = model_arenas t in
  let columns =
    match arenas with a :: _ -> List.map fst a.Model_arena.pools | [] -> []
  in
  J.obj
    [ ("type", J.str "model_arena_columns"); ("pools", J.list J.str columns) ]
  :: List.map
       (fun (a : Model_arena.t) ->
         J.obj
           [
             ("type", J.str "model_arena");
             ("model", J.str a.model);
             ("dialect", J.str a.dialect);
             ("kinds", J.int a.kinds);
             ("proven", J.bool a.proven);
             ("lower_bytes", J.opt J.i64 a.lower);
             ("upper_bytes", J.opt J.i64 a.upper);
             ("placed_bytes", J.opt J.i64 a.placed);
             ( "pools",
               J.list
                 (fun c -> J.opt J.i64 (Option.join (List.assoc_opt c a.pools)))
                 columns );
           ])
       arenas

(* The run's own grid, from its header: the largest budget and the first
   seed are the "with search" column. *)
let largest_budget t =
  match List.find_opt (is "header") t.rows with
  | None -> None
  | Some h -> (
      let config = J.field "config" h in
      let ints name =
        Option.bind config (J.field name)
        |> Fun.flip Option.bind J.to_list
        |> Option.value ~default:[] |> List.filter_map J.to_int64
      in
      match (ints "iterations", ints "seeds") with
      | (_ :: _ as its), s :: _ -> Some (List.fold_left Int64.max 0L its, s)
      | _ -> None)

let models_section buf t =
  match largest_budget t with
  | None -> ()
  | Some (top, seed) ->
      let strategies =
        [
          "greedy_by_area";
          "greedy_by_lifetime";
          "greedy_by_size";
          "greedy_by_size_best_fit";
        ]
      in
      let with_search s = Printf.sprintf "%s+improve@%Ld/%Ld" s top seed in
      let portfolio i = Printf.sprintf "portfolio@%Ld/%Ld" i seed in
      let short s =
        String.sub s 10 (String.length s - 10)
        (* drop "greedy_by_" *)
      in
      let mb = function
        | Some b -> Printf.sprintf "%.2f" (Int64.to_float b /. 1048576.)
        | None -> "?"
      in
      let arenas = model_arenas t in
      section buf
        (Printf.sprintf
           "Arena size per model (MiB; each strategy without search / with \
            %Ld             iterations, seed %Ld)"
           top seed);
      Buffer.add_string buf
        "Summed over element kinds. The minimum is proven unless it is \
         shown          as a range. A value equal to the minimum is the \
         smallest possible          arena for that node order.\n\n";
      table buf
        ([ "model"; "dialect"; "minimum" ]
        @ List.map short strategies @ [ "portfolio" ])
        (List.map
           (fun (a : Model_arena.t) ->
             let pool c = mb (Option.join (List.assoc_opt c a.pools)) in
             let minimum =
               if a.proven then mb a.lower else mb a.lower ^ "–" ^ mb a.upper
             in
             [ a.model; a.dialect; minimum ]
             @ List.map
                 (fun s -> pool s ^ " / " ^ pool (with_search s))
                 strategies
             @ [ pool (portfolio 0L) ^ " / " ^ pool (portfolio top) ])
           arenas)

let paired_section buf t =
  match t.dialects with
  | [ a; b ] ->
      let agg d =
        List.map (fun j -> (get_s "model" j, j)) (rows_of t "aggregate" d)
      in
      let aa = agg a and bb = agg b in
      let pairs =
        List.filter_map
          (fun (m, ja) ->
            Option.map (fun jb -> (m, ja, jb)) (List.assoc_opt m bb))
          aa
      in
      section buf
        (Printf.sprintf "Paired dialects (%d models evaluated in both, MiB)"
           (List.length pairs));
      let total f = List.fold_left (fun s x -> Int64.add s (f x)) 0L in
      table buf [ "figure"; a; b ]
        (List.map
           (fun (name, field) ->
             [
               name;
               mib (total (fun (_, ja, _) -> get_i64 field ja) pairs);
               mib (total (fun (_, _, jb) -> get_i64 field jb) pairs);
             ])
           [
             ("production pool", "production_bytes");
             ("placed pool (exact sizes)", "placed_bytes");
             ("best observed pool", "best_observed_bytes");
             ("proven lower bound", "lower_bytes");
             ("per-kind live bound", "per_kind_live_bound_bytes");
             ("combined live bound", "combined_live_bound_bytes");
             ("outside the arena", "out_of_arena_bytes");
           ])
  | _ -> ()

let hard_section buf t =
  let problems = Hashtbl.create 64 in
  List.iter
    (fun j -> if is "problem" j then Hashtbl.replace problems (key j) j)
    t.rows;
  let open_ =
    List.filter (fun j -> is "comparison" j && not (get_b "closed" j)) t.rows
    |> List.sort (fun a b ->
        Int64.compare
          (Int64.sub (get_i64 "upper" b) (get_i64 "lower" b))
          (Int64.sub (get_i64 "upper" a) (get_i64 "lower" a)))
  in
  section buf
    (Printf.sprintf "Unresolved minima (%d problems; largest gaps)"
       (List.length open_));
  table buf
    [
      "model";
      "dialect";
      "kind";
      "lower";
      "upper";
      "blocks";
      "conflicts";
      "peak live";
    ]
    (List.filteri (fun i _ -> i < 15) open_
    |> List.map (fun j ->
        let p = Hashtbl.find_opt problems (key j) in
        let pi name =
          Option.fold ~none:"?" ~some:(fun p -> string_of_int (get_i name p)) p
        in
        let m, d, k = key j in
        let o = Option.value ~default:"?" in
        [
          o m;
          o d;
          o k;
          Int64.to_string (get_i64 "lower" j);
          Int64.to_string (get_i64 "upper" j);
          pi "blocks";
          pi "conflicts";
          pi "peak_live_blocks";
        ]))

let cost_section buf t =
  section buf "Planning cost (seconds)";
  let sum ty field d =
    List.fold_left
      (fun a j -> a +. get_f field j)
      0.
      (match d with
      | None -> List.filter (is ty) t.rows
      | Some d -> rows_of t ty d)
  in
  Buffer.add_string buf
    (Printf.sprintf "- import and normalization, shared: %.2f\n"
       (sum "model" "import_s" None));
  List.iter
    (fun d ->
      Buffer.add_string buf
        (Printf.sprintf
           "- %s: conversion %.2f, dry run %.2f, allocators and reference %.2f\n"
           d
           (sum "dialect" "convert_s" (Some d))
           (sum "dialect" "dry_run_s" (Some d))
           (sum "dialect" "eval_s" (Some d))))
    t.dialects

let markdown t =
  let buf = Buffer.create 4096 in
  Buffer.add_string buf "# Arena allocator evaluation\n\n";
  header_section buf t;
  Buffer.add_string buf
    (Printf.sprintf "\nModels completed: %d/%d. Result: %s.\n" t.models
       t.expected
       (if ok t then "complete" else "INCOMPLETE"));
  coverage_section buf t;
  reference_section buf t;
  leaderboard_section buf t;
  paired_section buf t;
  hard_section buf t;
  cost_section buf t;
  models_section buf t;
  Buffer.contents buf
