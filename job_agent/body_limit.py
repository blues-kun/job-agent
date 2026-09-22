"""按实际ASGI请求字节数限制输入，不能靠省略Content-Length绕过。"""
from starlette.responses import JSONResponse


class BodyLimitMiddleware:
    def __init__(self,app):self.app=app

    async def __call__(self,scope,receive,send):
        if scope["type"]!="http" or scope.get("method") not in {"POST","PUT","PATCH","DELETE"}:
            return await self.app(scope,receive,send)
        maximum=2200000 if scope.get("path")=="/api/v2/document" else 150000
        body=bytearray()
        while True:
            message=await receive()
            if message["type"]=="http.disconnect":return
            body.extend(message.get("body",b""))
            if len(body)>maximum:
                return await JSONResponse({"detail":"请求实际大小超出上限"},status_code=413)(scope,receive,send)
            if not message.get("more_body",False):break
        delivered=False
        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered=True
                return {"type":"http.request","body":bytes(body),"more_body":False}
            return await receive()
        await self.app(scope,bounded_receive,send)
