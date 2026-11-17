open Err.Syntax

module Code_over_budget = struct
  type t = { budget : Code_bytes.t; needed : Code_bytes.t }
end

type error = [ Coral_layout.error | `Code_over_budget of Code_over_budget.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Code_over_budget { Code_over_budget.budget; needed } ->
      Fmt.pf ppf "code: %a bytes needed, %a budget" Code_bytes.pp needed
        Code_bytes.pp budget
  | (`Missing_region _ | `Over_capacity _) as e -> Coral_layout.pp_error ppf e

let budget (p : Coral_profile.t) =
  List.find_opt (fun (r : Coral_profile.Region.t) -> r.role = Itcm) p.regions
  |> Err.of_option (`Missing_region Coral_profile.Role.Itcm)
  |> Err.map (fun (r : Coral_profile.Region.t) -> Code_bytes.of_int64 r.size)

let check_code p needed =
  let* budget = budget p in
  if Code_bytes.compare needed budget > 0 then
    Err.fail (`Code_over_budget { Code_over_budget.budget; needed })
  else Err.return ()

module Placed = struct
  type t = {
    code : Code_bytes.t;
    layout : Coral_layout.t;
    slots : (string * Coral_layout.Placed_slot.t) list;
  }
end

let place p (image : Image.t) =
  let* () = check_code p image.code in
  let segment =
    {
      Coral_layout.Segment.align = 4L;
      executable = true;
      name = "text";
      size = Code_bytes.to_int64 image.code;
    }
  in
  let slots =
    List.map
      (fun (a : Image.Allocation.t) ->
        Coral_layout.Slot.make
          ~align:(Data_bytes.to_int64 a.align)
          a.placement
          (Data_bytes.to_int64 a.bytes))
      image.allocations
  in
  let* layout = Coral_layout.place p [ segment ] slots in
  Err.return
    {
      Placed.code = image.code;
      layout;
      (* [Coral_layout.place] returns one placed slot per slot, in order. *)
      slots =
        List.map2
          (fun (a : Image.Allocation.t) s -> (a.name, s))
          image.allocations layout.slots;
    }

module Admission = struct
  type rejection = { name : string; reason : error }
  type t = { placed : (string * Placed.t) list; rejected : rejection list }
end

let admit p images =
  let placed, rejected =
    List.fold_left
      (fun (placed, rejected) (image : Image.t) ->
        match Err.payload (place p image) with
        | Ok pl -> ((image.name, pl) :: placed, rejected)
        | Error reason ->
            (placed, { Admission.name = image.name; reason } :: rejected))
      ([], []) images
  in
  let placed = List.rev placed and rejected = List.rev rejected in
  if placed = [] then Err.fail (`None_fit rejected)
  else Err.return { Admission.placed; rejected }

module Candidate = struct
  type t = { census : Coral_estimate.Coral_census.t; image : Image.t }
end

module Selection = struct
  type t = {
    cycles : float;
    name : string;
    placed : Placed.t;
    ranked : (string * float) list;
    rejected : Admission.rejection list;
  }
end

let select p ~core ~rates (candidates : Candidate.t list) =
  let* adm = admit p (List.map (fun (c : Candidate.t) -> c.image) candidates) in
  let priced =
    List.filter_map
      (fun (c : Candidate.t) ->
        List.assoc_opt c.image.name adm.placed
        |> Option.map (fun placed ->
            let cycles =
              Coral_estimate.Coral_model.Rates.predict rates
                (Coral_estimate.Coral_model.features ~rates core c.census)
            in
            (c.image.name, cycles, placed)))
      candidates
  in
  let cmp (n1, c1, _) (n2, c2, _) =
    match Float.compare c1 c2 with 0 -> String.compare n1 n2 | k -> k
  in
  match List.stable_sort cmp priced with
  | [] -> Err.fail (`None_fit adm.rejected)
  | (name, cycles, placed) :: _ as sorted ->
      Err.return
        {
          Selection.cycles;
          name;
          placed;
          ranked = List.map (fun (n, c, _) -> (n, c)) sorted;
          rejected = adm.rejected;
        }
