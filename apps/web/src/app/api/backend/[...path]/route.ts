import { NextRequest, NextResponse } from "next/server";
import { API_BASE_URL } from "@/lib/config";

async function proxyRequest(req: NextRequest, params: { path: string[] }) {
  const token = req.cookies.get("stetho_token")?.value;
  const path = params.path.join("/");
  const searchParams = req.nextUrl.searchParams.toString();
  const queryString = searchParams ? `?${searchParams}` : "";
  const targetUrl = `${API_BASE_URL}/api/v1/${path}${queryString}`;

  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const contentType = req.headers.get("content-type");
  if (contentType && !contentType.includes("multipart/form-data")) {
    headers["Content-Type"] = contentType;
  }

  try {
    let body: any = null;
    if (req.method !== "GET" && req.method !== "HEAD") {
      if (contentType && contentType.includes("multipart/form-data")) {
        body = await req.formData();
      } else {
        body = await req.text();
      }
    }

    const res = await fetch(targetUrl, {
      method: req.method,
      headers,
      body,
      cache: "no-store",
    });

    const responseContentType = res.headers.get("content-type") || "";
    let data;
    if (responseContentType.includes("application/json")) {
      data = await res.json();
      return NextResponse.json(data, { status: res.status });
    } else {
      const blob = await res.blob();
      return new NextResponse(blob, {
        status: res.status,
        headers: {
          "Content-Type": responseContentType,
        },
      });
    }
  } catch (err: any) {
    return NextResponse.json(
      { detail: `Failed to connect to backend service: ${err.message}` },
      { status: 502 }
    );
  }
}

export async function GET(req: NextRequest, { params }: { params: { path: string[] } }) {
  return proxyRequest(req, params);
}

export async function POST(req: NextRequest, { params }: { params: { path: string[] } }) {
  return proxyRequest(req, params);
}

export async function PUT(req: NextRequest, { params }: { params: { path: string[] } }) {
  return proxyRequest(req, params);
}

export async function DELETE(req: NextRequest, { params }: { params: { path: string[] } }) {
  return proxyRequest(req, params);
}
