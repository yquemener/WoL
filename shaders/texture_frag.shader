uniform sampler2D texture;
varying mediump vec4 texc;
void main(void)
{
     vec4 texColor = texture2D(texture, texc.st);
     gl_FragColor=texColor;
}