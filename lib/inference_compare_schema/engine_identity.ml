module W = Wire
module Engine = W.Text (Id.Engine)
module Route = W.Text (Id.Engine_route)

type t = {
  engine : Id.Engine.t;
  route : Id.Engine_route.t;
  build_digest : Digest.Build.t;
  configuration_digest : Digest.Configuration.t;
  artifact_digest : Digest.Engine_artifact.t option;
  environment_digest : Digest.Environment.t;
}

let compare a b =
  let ( <?> ) c f = if c <> 0 then c else f () in
  Id.Engine.compare a.engine b.engine <?> fun () ->
  Id.Engine_route.compare a.route b.route <?> fun () ->
  Digest.Build.compare a.build_digest b.build_digest <?> fun () ->
  Digest.Configuration.compare a.configuration_digest b.configuration_digest
  <?> fun () ->
  Option.compare Digest.Engine_artifact.compare a.artifact_digest
    b.artifact_digest
  <?> fun () ->
  Digest.Environment.compare a.environment_digest b.environment_digest

let equal a b = compare a b = 0

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* engine = W.Obj.required o "engine" Engine.decode in
      let* route = W.Obj.required o "route" Route.decode in
      let* build_digest = W.Obj.required o "build_digest" Wire_digest.build in
      let* configuration_digest =
        W.Obj.required o "configuration_digest" Wire_digest.configuration
      in
      let* artifact_digest =
        W.Obj.optional o "artifact_digest" Wire_digest.engine_artifact
      in
      let* environment_digest =
        W.Obj.required o "environment_digest" Wire_digest.environment
      in
      Ok
        {
          engine;
          route;
          build_digest;
          configuration_digest;
          artifact_digest;
          environment_digest;
        })

let encode e =
  W.obj
    [
      W.field "engine" Engine.encode e.engine;
      W.field "route" Route.encode e.route;
      W.field "build_digest" Wire_digest.encode_build e.build_digest;
      W.field "configuration_digest" Wire_digest.encode_configuration
        e.configuration_digest;
      W.field "artifact_digest"
        (W.enc_option Wire_digest.encode_engine_artifact)
        e.artifact_digest;
      W.field "environment_digest" Wire_digest.encode_environment
        e.environment_digest;
    ]
