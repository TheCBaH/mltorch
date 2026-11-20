module Pass = Loop_ir.Loop_opt.Pass
module Numerics = Loop_ir.Loop_numerics

(* Each closed type's names come from one table, so [name], [of_name] and [all]
   cannot drift apart. *)
module type Names = sig
  type t

  val table : (t * string) list
end

module Closed (N : Names) = struct
  let all = List.map fst N.table
  let name v = List.assoc v N.table

  let of_name s =
    List.find_map
      (fun (v, n) -> if String.equal n s then Some v else None)
      N.table
end

module Dialect = struct
  type t = Native | Native4d

  include Closed (struct
    type nonrec t = t

    let table = [ (Native, "native"); (Native4d, "native4d") ]
  end)
end

module Metric = struct
  type t = Coral_cycles | Scalar_loops | Vector_loops

  include Closed (struct
    type nonrec t = t

    let table =
      [
        (Coral_cycles, "coral_cycles");
        (Scalar_loops, "scalar_loops");
        (Vector_loops, "vector_loops");
      ]
  end)
end

module Frontend = struct
  type t = Direct | Structured

  include Closed (struct
    type nonrec t = t

    let table = [ (Direct, "direct"); (Structured, "structured") ]
  end)
end

module Pipeline = struct
  type t = Exact | Planned | Representation

  include Closed (struct
    type nonrec t = t

    let table =
      [
        (Exact, "exact");
        (Planned, "planned");
        (Representation, "representation");
      ]
  end)
end

module Target = struct
  type t = Aarch64 | Riscv32 | Riscv32_grouped | X86_64

  include Closed (struct
    type nonrec t = t

    let table =
      [
        (Aarch64, "aarch64");
        (Riscv32, "riscv32");
        (Riscv32_grouped, "riscv32_grouped");
        (X86_64, "x86_64");
      ]
  end)
end

module Mir_stage = struct
  type t = Allocated | Generic | Realized | Scanned | Selected

  include Closed (struct
    type nonrec t = t

    let table =
      [
        (Allocated, "allocated");
        (Generic, "generic");
        (Realized, "realized");
        (Scanned, "scanned");
        (Selected, "selected");
      ]
  end)
end

module Coral_view = struct
  type t = Estimate | Rvv_kernel | Transfer_plan | Zvt_kernel

  include Closed (struct
    type nonrec t = t

    let table =
      [
        (Estimate, "estimate");
        (Rvv_kernel, "rvv_kernel");
        (Transfer_plan, "transfer_plan");
        (Zvt_kernel, "zvt_kernel");
      ]
  end)
end

module Loop_spec = struct
  type t = { passes : Pass.t list; numerics : Numerics.t option }

  (* A selection is a sub-sequence of the fixed pipeline, whatever order or
     repetition it was written in. *)
  let make ?numerics passes =
    { passes = List.filter (fun p -> List.mem p passes) Pass.all; numerics }

  let default = { passes = Pass.all; numerics = None }
end

module Ssa_spec = struct
  type t = {
    frontend : Frontend.t;
    pipeline : Pipeline.t;
    numerics : Numerics.t option;
  }

  let default = { frontend = Structured; pipeline = Planned; numerics = None }
end

module Route = struct
  type t = Loop of Loop_spec.t | Ssa of Ssa_spec.t

  let default = Loop Loop_spec.default
end

module Mir_spec = struct
  type t = { target : Target.t; stage : Mir_stage.t }

  let default = { target = X86_64; stage = Realized }
end

type family = Asm | C | Coral | Js | Loop | Machine_ir | Ssa | Wat

type variant =
  | Asm of Target.t
  | C of Route.t
  | Coral of Coral_view.t
  | Js of Route.t
  | Loop of Loop_spec.t
  | Machine_ir of Mir_spec.t
  | Ssa of Ssa_spec.t
  | Wat of Route.t

module Family = Closed (struct
  type t = family

  let table : (family * string) list =
    [
      (Asm, "asm");
      (C, "c");
      (Coral, "coral");
      (Js, "js");
      (Loop, "loop");
      (Machine_ir, "machine_ir");
      (Ssa, "ssa");
      (Wat, "wat");
    ]
end)

let reaches dialect (family : family) =
  match (dialect, family) with
  | Dialect.Native, _ -> true
  | Dialect.Native4d, (C | Js | Loop | Ssa | Wat) -> true
  | Dialect.Native4d, (Asm | Coral | Machine_ir) -> false

let families = Family.all
let family_name = Family.name
let family_of_name = Family.of_name

let family : variant -> family = function
  | Asm _ -> Asm
  | C _ -> C
  | Coral _ -> Coral
  | Js _ -> Js
  | Loop _ -> Loop
  | Machine_ir _ -> Machine_ir
  | Ssa _ -> Ssa
  | Wat _ -> Wat

let default : family -> variant = function
  | Asm -> Asm Target.X86_64
  | C -> C Route.default
  | Coral -> Coral Coral_view.Estimate
  | Js -> Js Route.default
  | Loop -> Loop Loop_spec.default
  | Machine_ir -> Machine_ir Mir_spec.default
  | Ssa -> Ssa Ssa_spec.default
  | Wat -> Wat Route.default

module Parse_error = struct
  type t =
    | Duplicate_segment of { family : family; segment : string }
    | Empty_segment of family
    | Unknown_family of string
    | Unknown_segment of { family : family; segment : string }

  let pp ppf = function
    | Duplicate_segment { family; segment } ->
        Fmt.pf ppf "%s: segment %S repeats an axis already given"
          (family_name family) segment
    | Empty_segment family ->
        Fmt.pf ppf "%s: empty segment" (family_name family)
    | Unknown_family s -> Fmt.pf ppf "unknown family %S" s
    | Unknown_segment { family; segment } ->
        Fmt.pf ppf "%s: unknown segment %S" (family_name family) segment
end

(* ---- naming ---- *)

let passes_segment passes =
  if passes = Pass.all then None
  else if passes = [] then Some "raw"
  else Some (String.concat "," (List.map Pass.name passes))

let numerics_segment = Option.map Numerics.name

let loop_segments (s : Loop_spec.t) =
  List.filter_map Fun.id
    [ passes_segment s.passes; numerics_segment s.numerics ]

let ssa_segments (s : Ssa_spec.t) =
  List.filter_map Fun.id
    [
      (if s.frontend = Ssa_spec.default.frontend then None
       else Some (Frontend.name s.frontend));
      (if s.pipeline = Ssa_spec.default.pipeline then None
       else Some (Pipeline.name s.pipeline));
      numerics_segment s.numerics;
    ]

let route_segments : Route.t -> string list = function
  | Loop s when s = Loop_spec.default -> []
  | Loop s -> "loop" :: loop_segments s
  | Ssa s -> "ssa" :: ssa_segments s

let segments : variant -> string list = function
  | Asm t -> if t = Target.X86_64 then [] else [ Target.name t ]
  | C r | Js r | Wat r -> route_segments r
  | Coral v -> if v = Estimate then [] else [ Coral_view.name v ]
  | Loop s -> loop_segments s
  | Machine_ir m ->
      List.filter_map Fun.id
        [
          (if m.target = Mir_spec.default.target then None
           else Some (Target.name m.target));
          (if m.stage = Mir_spec.default.stage then None
           else Some (Mir_stage.name m.stage));
        ]
  | Ssa s -> ssa_segments s

let name v =
  let f = family_name (family v) in
  match segments v with [] -> f | segs -> f ^ ":" ^ String.concat "/" segs

let pp ppf v = Fmt.string ppf (name v)

(* ---- parsing ---- *)

type 'a step = Next of 'a | Duplicate | Unknown

let fold_segments fam init step segs =
  let rec go acc = function
    | [] -> Ok acc
    | "" :: _ -> Error (Parse_error.Empty_segment fam)
    | seg :: rest -> (
        match step acc seg with
        | Next acc -> go acc rest
        | Duplicate ->
            Error
              (Parse_error.Duplicate_segment { family = fam; segment = seg })
        | Unknown ->
            Error (Parse_error.Unknown_segment { family = fam; segment = seg }))
  in
  go init segs

let pass_list seg =
  match seg with
  | "all" -> Some Pass.all
  | "raw" -> Some []
  | _ ->
      let names = String.split_on_char ',' seg in
      List.fold_right
        (fun n acc ->
          match (acc, Pass.of_name n) with
          | Some l, Some p -> Some (p :: l)
          | _ -> None)
        names (Some [])

(* A segment of the Loop spec: a pass selection or a numerics policy. Their
   names are disjoint, so the axis is read off the name. *)
let loop_step fam (passes, numerics) seg =
  match Numerics.of_name seg with
  | Some n -> if numerics <> None then Duplicate else Next (passes, Some n)
  | None -> (
      match pass_list seg with
      | Some p -> if passes <> None then Duplicate else Next (Some p, numerics)
      | None ->
          ignore fam;
          Unknown)

let loop_spec fam segs =
  Result.map
    (fun (passes, numerics) ->
      Loop_spec.make ?numerics (Option.value passes ~default:Pass.all))
    (fold_segments fam (None, None) (loop_step fam) segs)

let ssa_step (f, p, n) seg =
  match Frontend.of_name seg with
  | Some v -> if f <> None then Duplicate else Next (Some v, p, n)
  | None -> (
      match Pipeline.of_name seg with
      | Some v -> if p <> None then Duplicate else Next (f, Some v, n)
      | None -> (
          match Numerics.of_name seg with
          | Some v -> if n <> None then Duplicate else Next (f, p, Some v)
          | None -> Unknown))

let ssa_spec fam segs =
  Result.map
    (fun (f, p, numerics) : Ssa_spec.t ->
      {
        frontend = Option.value f ~default:Ssa_spec.default.frontend;
        pipeline = Option.value p ~default:Ssa_spec.default.pipeline;
        numerics;
      })
    (fold_segments fam (None, None, None) ssa_step segs)

let route fam segs : (Route.t, Parse_error.t) result =
  match segs with
  | [] -> Ok Route.default
  | "" :: _ -> Error (Parse_error.Empty_segment fam)
  | "loop" :: rest -> Result.map (fun s -> Route.Loop s) (loop_spec fam rest)
  | "ssa" :: rest -> Result.map (fun s -> Route.Ssa s) (ssa_spec fam rest)
  | seg :: _ ->
      Error (Parse_error.Unknown_segment { family = fam; segment = seg })

let single fam of_name segs ~default =
  match segs with
  | [] -> Ok default
  | [ "" ] -> Error (Parse_error.Empty_segment fam)
  | [ seg ] -> (
      match of_name seg with
      | Some v -> Ok v
      | None ->
          Error (Parse_error.Unknown_segment { family = fam; segment = seg }))
  | _ :: seg :: _ ->
      Error (Parse_error.Unknown_segment { family = fam; segment = seg })

let mir_step (t, s) seg =
  match Target.of_name seg with
  | Some v -> if t <> None then Duplicate else Next (Some v, s)
  | None -> (
      match Mir_stage.of_name seg with
      | Some v -> if s <> None then Duplicate else Next (t, Some v)
      | None -> Unknown)

let of_name s : (variant, Parse_error.t) result =
  let fam_s, segs =
    match String.index_opt s ':' with
    | None -> (s, [])
    | Some i ->
        ( String.sub s 0 i,
          String.split_on_char '/'
            (String.sub s (i + 1) (String.length s - i - 1)) )
  in
  match family_of_name fam_s with
  | None -> Error (Parse_error.Unknown_family fam_s)
  | Some fam -> (
      let ( let* ) = Result.bind in
      match fam with
      | Asm ->
          let* t = single fam Target.of_name segs ~default:Target.X86_64 in
          Ok (Asm t)
      | C -> Result.map (fun r -> C r) (route fam segs)
      | Js -> Result.map (fun r -> Js r) (route fam segs)
      | Wat -> Result.map (fun r -> Wat r) (route fam segs)
      | Coral ->
          let* v =
            single fam Coral_view.of_name segs ~default:Coral_view.Estimate
          in
          Ok (Coral v)
      | Loop ->
          if segs = [] then Ok (Loop Loop_spec.default)
          else Result.map (fun s -> Loop s) (loop_spec fam segs)
      | Ssa ->
          if segs = [] then Ok (Ssa Ssa_spec.default)
          else Result.map (fun s -> Ssa s) (ssa_spec fam segs)
      | Machine_ir ->
          let* t, st = fold_segments fam (None, None) mir_step segs in
          Ok
            (Machine_ir
               {
                 target = Option.value t ~default:Mir_spec.default.target;
                 stage = Option.value st ~default:Mir_spec.default.stage;
               }))

module Node_id = Graph_ir.Node_id

type t = { dialect : Dialect.t; node : Node_id.t; variant : variant }

let make ?(dialect = Dialect.Native) node variant = { dialect; node; variant }

let equal a b =
  a.dialect = b.dialect && Node_id.equal a.node b.node && a.variant = b.variant

let pp_key ppf k =
  Fmt.pf ppf "%s/%a/%a" (Dialect.name k.dialect) Node_id.pp k.node pp k.variant
