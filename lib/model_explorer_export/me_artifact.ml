(* The lowering inspector's answer. See the .mli. *)

module Tag = struct
  type t = { kind : string; data : string list }

  let jsont =
    Jsont.Object.map ~kind:"artifactTag" (fun kind data -> { kind; data })
    |> Jsont.Object.mem "kind" Jsont.string ~enc:(fun t -> t.kind)
    |> Jsont.Object.mem "data" (Jsont.list Jsont.string) ~enc:(fun t -> t.data)
    |> Jsont.Object.finish
end

module Body = struct
  (* @alphabetical *)
  type t =
    | Refused of Tag.t
    | Text of { language : string; text : string; truncated : bool }

  let jsont =
    Jsont.Object.map ~kind:"artifactBody"
      (fun kind language text truncated refusal ->
        let need what = function
          | Some v -> v
          | None ->
              Jsont.Error.msgf Jsont.Meta.none
                "artifact body %S does not carry its own %s" kind what
        in
        match kind with
        | "refused" -> Refused (need "refusal" refusal)
        | "text" ->
            Text
              {
                language = need "language" language;
                text = need "text" text;
                truncated = need "truncated" truncated;
              }
        | _ -> Jsont.Error.msgf Jsont.Meta.none "unknown artifact body %S" kind)
    |> Jsont.Object.mem "kind" Jsont.string ~enc:(function
      | Refused _ -> "refused"
      | Text _ -> "text")
    |> Jsont.Object.opt_mem "language" Jsont.string ~enc:(function
      | Text t -> Some t.language
      | Refused _ -> None)
    |> Jsont.Object.opt_mem "text" Jsont.string ~enc:(function
      | Text t -> Some t.text
      | Refused _ -> None)
    |> Jsont.Object.opt_mem "truncated" Jsont.bool ~enc:(function
      | Text t -> Some t.truncated
      | Refused _ -> None)
    |> Jsont.Object.opt_mem "refusal" Tag.jsont ~enc:(function
      | Refused r -> Some r
      | Text _ -> None)
    |> Jsont.Object.finish
end

type t = { key : Lowering_key.t; body : Body.t; notes : Tag.t list }

(* A cut inside a multi-byte sequence would leave bytes the JSON writer cannot
   encode, so it backs up to the start of the sequence: a continuation byte is
   10xxxxxx. The scan is at most three bytes. *)
let text ~limits ~language text =
  let max = limits.Me_limits.Limits.max_artifact_bytes in
  if String.length text <= max then
    Body.Text { language; text; truncated = false }
  else begin
    let cut = ref max in
    while !cut > 0 && Char.code text.[!cut] land 0xC0 = 0x80 do
      decr cut
    done;
    Body.Text { language; text = String.sub text 0 !cut; truncated = true }
  end

let jsont =
  Jsont.Object.map ~kind:"artifact" (fun key body notes -> { key; body; notes })
  |> Jsont.Object.mem "key" Me_request.Lowering_wire.jsont ~enc:(fun t -> t.key)
  |> Jsont.Object.mem "body" Body.jsont ~enc:(fun t -> t.body)
  |> Jsont.Object.mem "notes" (Jsont.list Tag.jsont) ~enc:(fun t -> t.notes)
  |> Jsont.Object.finish
