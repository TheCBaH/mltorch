module Item = Rv32_kernel_item
module Label = Rv32_kernel_label
module At = Item.At
module Label_map = Map.Make (Label)
module String_map = Map.Make (String)

module Name_kind = struct
  type t = Export | Kernel | Label

  let pp ppf = function
    | Export -> Fmt.string ppf "export"
    | Kernel -> Fmt.string ppf "kernel"
    | Label -> Fmt.string ppf "label"
end

module Invalid = struct
  type t =
    | Alignment of { at : At.t; boundary : int }
    | Duplicate_export of { name : string; first : At.t; second : At.t }
    | Duplicate_label of { label : Label.t; first : At.t; second : At.t }
    | Export_twice of { label : Label.t; first : At.t; second : At.t }
    | Export_undefined of { name : string; label : Label.t; at : At.t }
    | Immediate_range of { at : At.t; range : Rv32_kernel_imm.Imm_range.t }
    | Invalid_name of { kind : Name_kind.t; name : string; at : At.t option }
    | Negative_zero of { at : At.t; length : int }
    | Unresolved_label of { label : Label.t; at : At.t }
    | Vsetvli_keeps_vl of { at : At.t }

  (* Where to report it: the later of two clashing items. *)
  let at = function
    | Alignment { at; _ }
    | Duplicate_export { second = at; _ }
    | Duplicate_label { second = at; _ }
    | Export_twice { second = at; _ }
    | Export_undefined { at; _ }
    | Immediate_range { at; _ }
    | Negative_zero { at; _ }
    | Unresolved_label { at; _ }
    | Vsetvli_keeps_vl { at } ->
        Some at
    | Invalid_name { at; _ } -> at

  let pp ppf = function
    | Alignment { at; boundary } ->
        Fmt.pf ppf "%a: alignment %d is not a positive power of two" At.pp at
          boundary
    | Duplicate_export { name; first; second } ->
        Fmt.pf ppf "%a: export %s is already exported at %a" At.pp second name
          At.pp first
    | Duplicate_label { label; first; second } ->
        Fmt.pf ppf "%a: label %a is already defined at %a" At.pp second Label.pp
          label At.pp first
    | Export_twice { label; first; second } ->
        Fmt.pf ppf "%a: label %a already has an export name, at %a" At.pp second
          Label.pp label At.pp first
    | Export_undefined { name; label; at } ->
        Fmt.pf ppf "%a: export %s names label %a, which is never defined" At.pp
          at name Label.pp label
    | Immediate_range { at; range } ->
        Fmt.pf ppf "%a: %a" At.pp at Rv32_kernel_imm.Imm_range.pp range
    | Negative_zero { at; length } ->
        Fmt.pf ppf "%a: a zero fill of %d bytes" At.pp at length
    | Invalid_name { kind; name; at } ->
        Fmt.pf ppf "%a%a name %S is not [A-Za-z_][A-Za-z0-9_]*"
          Fmt.(option (fun ppf at -> pf ppf "%a: " At.pp at))
          at Name_kind.pp kind name
    | Unresolved_label { label; at } ->
        Fmt.pf ppf "%a: label %a is never defined" At.pp at Label.pp label
    | Vsetvli_keeps_vl { at } ->
        Fmt.pf ppf
          "%a: vsetvli with rd = rs1 = x0 keeps vl; VLMAX needs a destination"
          At.pp at
end

type error = [ `Invalid of Invalid.t | `Kernel_invalid of Invalid.t list ]

let pp_error ppf : error -> unit = function
  | `Invalid i -> Invalid.pp ppf i
  | `Kernel_invalid is -> Fmt.(list ~sep:cut Invalid.pp |> vbox) ppf is

type t = { name : string; items : Item.t list }

let items k = k.items
let name k = k.name

let valid_name s =
  let first c = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c = '_' in
  let rest c = first c || (c >= '0' && c <= '9') in
  s <> "" && first s.[0] && String.for_all rest s

(* Each item with its position and the annotation in effect there. *)
let located items =
  let _, _, rev =
    List.fold_left
      (fun (index, note, acc) item ->
        let note' = match item with Item.Note n -> n | _ -> note in
        (index + 1, note', (item, { At.index; note }) :: acc))
      (0, None, []) items
  in
  List.rev rev

let first_by_label located =
  List.fold_left
    (fun acc (item, at) ->
      match item with
      | Item.Label l when not (Label_map.mem l acc) -> Label_map.add l at acc
      | _ -> acc)
    Label_map.empty located

let duplicate_labels located =
  let _, rev =
    List.fold_left
      (fun (seen, acc) (item, second) ->
        match item with
        | Item.Label label -> (
            match Label_map.find_opt label seen with
            | None -> (Label_map.add label second seen, acc)
            | Some first ->
                (seen, Invalid.Duplicate_label { label; first; second } :: acc))
        | _ -> (seen, acc))
      (Label_map.empty, []) located
  in
  List.rev rev

let export_clashes located =
  let _, _, rev =
    List.fold_left
      (fun (names, labels, acc) (item, second) ->
        match item with
        | Item.Export { name; label } ->
            let acc =
              match String_map.find_opt name names with
              | Some first ->
                  Invalid.Duplicate_export { name; first; second } :: acc
              | None -> acc
            in
            let acc =
              match Label_map.find_opt label labels with
              | Some first ->
                  Invalid.Export_twice { label; first; second } :: acc
              | None -> acc
            in
            ( String_map.update name
                (function None -> Some second | s -> s)
                names,
              Label_map.update label
                (function None -> Some second | s -> s)
                labels,
              acc )
        | _ -> (names, labels, acc))
      (String_map.empty, Label_map.empty, [])
      located
  in
  List.rev rev

(* Every label an item refers to, with the item. *)
let references located =
  List.concat_map
    (fun (item, at) ->
      match item with
      | Item.Insn i ->
          List.map (fun l -> (`Branch l, at)) (Rv32_kernel_insn.targets i)
      | Item.Export { name; label } -> [ (`Export (name, label), at) ]
      | _ -> [])
    located

let undefined_references defined located =
  List.filter_map
    (fun (reference, at) ->
      match reference with
      | `Branch label when not (Label_map.mem label defined) ->
          Some (Invalid.Unresolved_label { label; at })
      | `Export (name, label) when not (Label_map.mem label defined) ->
          Some (Invalid.Export_undefined { name; label; at })
      | _ -> None)
    (references located)

(* A bad name is reported once, at the first item that uses it. *)
let invalid_names ~name located =
  let used =
    List.concat_map
      (fun (item, at) ->
        match item with
        | Item.Label l -> [ (Name_kind.Label, Label.name l, at) ]
        | Item.Export { name; label } ->
            [
              (Name_kind.Export, name, at);
              (Name_kind.Label, Label.name label, at);
            ]
        | Item.Insn i ->
            List.map
              (fun l -> (Name_kind.Label, Label.name l, at))
              (Rv32_kernel_insn.targets i)
        | _ -> [])
      located
  in
  let kernel =
    if valid_name name then []
    else [ Invalid.Invalid_name { kind = Name_kind.Kernel; name; at = None } ]
  in
  let _, rev =
    List.fold_left
      (fun (seen, acc) (kind, name, at) ->
        if valid_name name || List.mem (kind, name) seen then (seen, acc)
        else
          ( (kind, name) :: seen,
            Invalid.Invalid_name { kind; name; at = Some at } :: acc ))
      ([], []) used
  in
  kernel @ List.rev rev

let bad_forms located =
  List.concat_map
    (fun (item, at) ->
      match item with
      | Item.Align boundary
        when boundary < 1 || boundary land (boundary - 1) <> 0 ->
          [ Invalid.Alignment { at; boundary } ]
      | Item.Zero length when length < 0 ->
          [ Invalid.Negative_zero { at; length } ]
      | Item.Insn i -> (
          List.map
            (fun range -> Invalid.Immediate_range { at; range })
            (Rv32_kernel_insn.out_of_range i)
          @
          match i with
          | Rv32_kernel_insn.Vsetvli
              { avl = Rv32_kernel_vtype.Avl.Vlmax { vl }; _ }
            when Rv32_kernel_reg.Gpr.equal vl Rv32_kernel_reg.Gpr.zero ->
              [ Invalid.Vsetvli_keeps_vl { at } ]
          | _ -> [])
      | _ -> [])
    located

let index_of i = match Invalid.at i with None -> -1 | Some at -> at.At.index

let finalize ~name items =
  let located = located items in
  let defined = first_by_label located in
  let invalid =
    invalid_names ~name located
    @ duplicate_labels located @ export_clashes located
    @ undefined_references defined located
    @ bad_forms located
    |> List.stable_sort (fun a b -> Int.compare (index_of a) (index_of b))
  in
  Err.Accum.all ~pos:__POS__ (List.map (fun i -> Err.fail i) invalid)
  |> Err.Accum.fold_errors (fun errors ->
      match List.map Err.Error.kind errors with
      | [ one ] -> `Invalid one
      | many -> `Kernel_invalid many)
  |> Err.map (fun (_ : unit list) -> { name; items })
